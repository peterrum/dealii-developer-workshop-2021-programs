#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/data_out_dof_data.templates.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

template <int dim, int patch_dim, int spacedim>
class DataOutResample
  : public DataOut_DoFData<dim, patch_dim, spacedim, spacedim>
{
public:
  DataOutResample(const Triangulation<patch_dim, spacedim> &tria,
                  const Mapping<patch_dim, spacedim> &      patch_mapping);

  void
  update_mapping(const Mapping<dim, spacedim> &mapping,
                 const unsigned int            n_subdivisions = 0);

  void
  build_patches();

protected:
  virtual const std::vector<::DataOutBase::Patch<patch_dim, spacedim>> &
  get_patches() const override;

private:
  const Triangulation<patch_dim, spacedim> &tria;
  DoFHandler<patch_dim, spacedim>           dof_handler_patch;
  const Mapping<patch_dim, spacedim> &      patch_mapping;


  Utilities::MPI::RemotePointEvaluation<dim, spacedim> rpe;
  std::shared_ptr<Utilities::MPI::Partitioner>         partitioner;
  std::vector<types::global_dof_index>                 indices;
  SmartPointer<const Mapping<dim, spacedim>>           mapping;


  DataOut<patch_dim, spacedim> data_out;
};



template <int dim, int patch_dim, int spacedim>
DataOutResample<dim, patch_dim, spacedim>::DataOutResample(
  const Triangulation<patch_dim, spacedim> &tria,
  const Mapping<patch_dim, spacedim> &      patch_mapping)
  : tria(tria)
  , dof_handler_patch(tria)
  , patch_mapping(patch_mapping)
{}



template <int dim, int patch_dim, int spacedim>
void
DataOutResample<dim, patch_dim, spacedim>::update_mapping(
  const Mapping<dim, spacedim> &mapping,
  const unsigned int            n_subdivisions)
{
  this->mapping = &mapping;

  FE_Q<patch_dim, spacedim> fe(std::max<unsigned int>(1, n_subdivisions));
  dof_handler_patch.distribute_dofs(fe);

  std::vector<Point<spacedim>>                                     points;
  std::vector<std::pair<types::global_dof_index, Point<spacedim>>> points_all;

  QGaussLobatto<patch_dim> quadrature_gl(fe.degree + 1);

  std::vector<Point<patch_dim>> quadrature_points;
  for (const auto i :
       FETools::hierarchic_to_lexicographic_numbering<patch_dim>(fe.degree))
    quadrature_points.push_back(quadrature_gl.point(i));
  Quadrature<patch_dim> quadrature(quadrature_points);

  FEValues<patch_dim, spacedim> fe_values(patch_mapping,
                                          fe,
                                          quadrature,
                                          update_quadrature_points);

  std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

  IndexSet index_set;
  DoFTools::extract_locally_active_dofs(dof_handler_patch, index_set);
  partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(index_set, MPI_COMM_WORLD);

  for (const auto &cell : dof_handler_patch.active_cell_iterators())
    {
      fe_values.reinit(cell);

      points.insert(points.end(),
                    fe_values.get_quadrature_points().begin(),
                    fe_values.get_quadrature_points().end());

      cell->get_dof_indices(dof_indices);

      for (const auto i : dof_indices)
        indices.push_back(partitioner->global_to_local(i));
    }

  std::sort(points_all.begin(),
            points_all.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  points_all.erase(std::unique(points_all.begin(),
                               points_all.end(),
                               [](const auto &a, const auto &b) {
                                 return a.first == b.first;
                               }),
                   points_all.end());

  for (const auto i : points_all)
    {
      indices.push_back(i.first);
      points.push_back(i.second);
    }

  rpe.reinit(points, *this->triangulation, *this->mapping);
}



template <int dim, int patch_dim, int spacedim>
void
DataOutResample<dim, patch_dim, spacedim>::build_patches()
{
  data_out.clear();

  if (rpe.is_ready() == false)
    update_mapping(*this->mapping, dof_handler_patch.get_fe().degree);

  std::vector<std::shared_ptr<LinearAlgebra::distributed::Vector<double>>>
    vectors;

  data_out.attach_dof_handler(dof_handler_patch);

  unsigned int counter = 0;

  for (const auto &i : this->dof_data)
    {
      const auto temp = dynamic_cast<
        internal::DataOutImplementation::
          DataEntry<dim, spacedim, LinearAlgebra::distributed::Vector<double>>
            *>(i.get());

      Assert(temp, ExcNotImplemented());

      const auto &dh = *temp->dof_handler;

      AssertDimension(dh.get_fe_collection().n_components(), 1);

      const auto values = VectorTools::point_values<1>(rpe, dh, *temp->vector);

      vectors.emplace_back(
        std::make_shared<LinearAlgebra::distributed::Vector<double>>(
          partitioner));

      for (unsigned int j = 0; j < values.size(); ++j)
        vectors.back()->local_element(indices[j]) = values[j];

      data_out.add_data_vector(
        *vectors.back(),
        std::string("temp_" + std::to_string(counter++)),
        DataOut_DoFData<patch_dim, patch_dim, spacedim, spacedim>::
          DataVectorType::type_dof_data);
    }

  data_out.build_patches(patch_mapping, dof_handler_patch.get_fe().degree);
}



template <int dim, int patch_dim, int spacedim>
const std::vector<::DataOutBase::Patch<patch_dim, spacedim>> &
DataOutResample<dim, patch_dim, spacedim>::get_patches() const
{
  return data_out.get_patches();
}



template <int dim>
class AnalyticalFunction : public Function<dim>
{
public:
  AnalyticalFunction()
    : Function<dim>(1)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const
  {
    (void)component;

    return p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
  }
};


template <int dim, int spacedim>
std::shared_ptr<const Utilities::MPI::Partitioner>
create_partitioner(const DoFHandler<dim, spacedim> &dof_handler)
{
  IndexSet locally_relevant_dofs;

  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  return std::make_shared<const Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    locally_relevant_dofs,
    dof_handler.get_communicator());
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const int dim       = 3;
  const int patch_dim = 2;
  const int spacedim  = 3;

  const unsigned int n_refinements_1 = 3;
  const unsigned int n_refinements_2 = 3;
  const MPI_Comm     comm            = MPI_COMM_WORLD;

  Triangulation<patch_dim, spacedim> tria_slice;
  GridGenerator::hyper_cube(tria_slice, -1.0, +1.0);
  tria_slice.refine_global(n_refinements_2);

  MappingQ1<patch_dim, spacedim> mapping_slice;

  Triangulation<dim, spacedim> tria_backround;
  GridGenerator::hyper_cube(tria_backround, -1.0, +1.0);
  tria_backround.refine_global(n_refinements_1);

  DoFHandler<dim, spacedim> dof_handler(tria_backround);
  dof_handler.distribute_dofs(FE_Q<dim, spacedim>{1});

  MappingQ1<dim, spacedim> mapping;

  LinearAlgebra::distributed::Vector<double> vector(
    create_partitioner(dof_handler));

  VectorTools::interpolate(mapping,
                           dof_handler,
                           AnalyticalFunction<dim>(),
                           vector);

  DataOutResample<dim, patch_dim, spacedim> data_out(tria_slice, mapping_slice);
  data_out.add_data_vector(dof_handler, vector, "solution_0");
  data_out.add_data_vector(dof_handler, vector, "solution_1");
  data_out.update_mapping(mapping);
  data_out.build_patches();
  data_out.write_vtu_with_pvtu_record("./", "data_out_01", 0, comm, 1, 1);
}
