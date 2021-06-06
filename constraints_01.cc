#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>

using namespace dealii;

/**
 * mpirun -np 4 ./constraints_01
 *
 * 6-----7-----8
 * | 0/2 | 1/3 |
 * 2-----3-----5
 * | 0/0 | 0/1 |  ... with dof-index, material-id/rank
 * 0-----1-----4
 *
 * Expected output: {3}, {3, 5}, {3, 7}, {3, 5, 7}
 * Actual output:   {},  {3, 5}, {3, 7}, {3, 5, 7}
 *
 * ... code to reproduce https://github.com/dealii/dealii/issues/11725
 */

template <typename Number>
IndexSet
collect_lines(const AffineConstraints<Number> &constraints,
              const unsigned int               size)
{
  IndexSet lines_local(size);
  for (const auto &line : constraints.get_lines())
    lines_local.add_index(line.index);
  return lines_local;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const int dim = 2;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(tria, 2);

  DoFHandler dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(1));

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->center()[0] > 0.5 && cell->center()[1] > 0.5)
      cell->set_material_id(1);

  AffineConstraints<double> constraints;

  std::vector<types::global_dof_index> dof_indices(
    dof_handler.get_fe().n_dofs_per_face());

  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const auto face : cell->face_indices())
      if (cell->is_locally_owned() && !cell->at_boundary(face) &&
          cell->material_id() != cell->neighbor(face)->material_id())
        {
          cell->face(face)->get_dof_indices(dof_indices);
          for (const auto i : dof_indices)
            constraints.add_line(i);
        }

  const auto all_constraints = Utilities::MPI::all_gather(
    MPI_COMM_WORLD, collect_lines(constraints, dof_handler.n_dofs()));

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    for (const auto i : all_constraints)
      i.print(std::cout);
}
