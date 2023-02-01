import os
import vedo


class Renderer:
    def __init__(
        self,
        pos_receiver,
        new_experiment_callback,
        clusters,
        pos_orig,
        pos_comp,
    ):
        self.pos_receiver = pos_receiver
        self.new_experiment_callback = new_experiment_callback
        self.plt = vedo.Plotter()
        self.generate_static_meshes()
        self.generate_dynamic_meshes(clusters, pos_orig, pos_comp)
        self.plt.add_button(
            self.show_clusters,
            pos=(0.5, 0.1),
            states=("show reflection clusters", "n"),
        )
        self.plt.add_button(
            self.show_normals,
            pos=(0.5, 0.05),
            states=("show wall normals", "n"),
        )
        self.plt.add_button(
            self.show_positions,
            pos=(0.5, 0),
            states=("show sender positions", "n"),
        )
        self.plt.add_button(
            self.run_new_experiment,
            pos=(0.1, 0.95),
            states=("New experiment", "n"),
        )
        self.plt.add(
            self.mesh_room,
            self.mesh_axes,
            self.mesh_clusters,
            self.mesh_normals,
            self.mesh_pos_orig,
            self.mesh_pos_comp,
        )
        self.plt.show(
            camera={
                "pos": (5, 5, 5),
                "viewup": (0, 0, 1),
                "focal_point": (0, 0, 0),
            },
        )

    def run_new_experiment(self):
        clusters, pos_orig, pos_comp = self.new_experiment_callback()
        self.plot_new_experiment(clusters, pos_orig, pos_comp)

    def plot_new_experiment(self, clusters, pos_orig, pos_comp):
        self.plt.remove(
            self.mesh_clusters,
            self.mesh_normals,
            self.mesh_pos_orig,
            self.mesh_pos_comp,
        )
        self.generate_dynamic_meshes(clusters, pos_orig, pos_comp)
        self.plt.add(
            self.mesh_clusters,
            self.mesh_normals,
            self.mesh_pos_comp,
            self.mesh_pos_comp,
        )

    def generate_static_meshes(self):
        self.mesh_room = vedo.Mesh(
            os.getcwd() + "/evaluation/testrooms/models/cube.obj"
        ).wireframe()
        self.mesh_axes = vedo.Axes(
            self.mesh_room,
            xrange=(-1.5, 1.5),
            yrange=(-1.5, 1.5),
            zrange=(-1.5, 1.5),
        )

    def generate_dynamic_meshes(self, clusters, pos_orig, pos_comp):
        self.mesh_clusters = [
            [
                vedo.Arrow(
                    start_pt=self.pos_receiver,
                    end_pt=(self.pos_receiver + reflection.direction),
                    s=0.002,
                    c=vedo.colors.color_map(
                        i, name="jet", vmin=0, vmax=len(clusters)
                    ),
                ).alpha(0)
                for reflection in cluster.reflected_signals
            ]
            for i, cluster in enumerate(clusters)
        ]
        self.mesh_normals = [
            vedo.Arrow(
                start_pt=self.pos_receiver,
                end_pt=(self.pos_receiver + cluster.wall_normal),
                s=0.002,
                c="green",
            ).alpha(0)
            for cluster in clusters
        ]
        self.mesh_pos_orig = [
            vedo.Point(position_original, c="red")
            for position_original in pos_orig
        ]
        self.mesh_pos_comp = [
            vedo.Point(position_computed, c="blue")
            for position_computed in pos_comp
        ]

    # Functions for visualization
    def show_clusters(self):
        # hide other meshes
        for pos_o in self.mesh_pos_orig:
            pos_o.alpha(0)
        for pos_c in self.mesh_pos_comp:
            pos_c.alpha(0)
        for normal in self.mesh_normals:
            normal.alpha(0)
        # Show clusters
        for cluster in self.mesh_clusters:
            for arrow in cluster:
                arrow.alpha(1)

    def show_normals(self):
        # hide other meshes
        for cluster in self.mesh_clusters:
            for arrow in cluster:
                arrow.alpha(0)
        for pos_o in self.mesh_pos_orig:
            pos_o.alpha(0)
        for pos_c in self.mesh_pos_comp:
            pos_c.alpha(0)
        # show normal meshes
        for normal in self.mesh_normals:
            normal.alpha(1)
        print("showing normals")

    def show_positions(self):
        # hide other meshes
        for cluster in self.mesh_clusters:
            for arrow in cluster:
                arrow.alpha(0)
        for normal in self.mesh_normals:
            normal.alpha(0)
        # show positions
        for pos_o in self.mesh_pos_orig:
            pos_o.alpha(1)
        for pos_c in self.mesh_pos_comp:
            pos_c.alpha(1)
