""" Main vox2usd conversion logic

This module contains Vox2UsdConverter class which serves as the entry
point for the script and has most of the conversion logic.

TODO:
    * Figure out MV instancing (Ref)
    * Should I rotate studs to always point up?
    * Compute bboxes
"""

import os

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt
from vox2usd._meshing import GreedyMeshing
from vox2usd.constants import (
    DEFAULT_METERS_PER_VOXEL,
    GEOMETRY_SCOPE_NAME,
    LOOKS_SCOPE_NAME,
    GeometryVariantSetNames,
    PointShapeVariantSetNames,
    ShaderVariantSetNames,
    StudMesh,
)
from vox2usd.vox import VoxBaseMaterial, VoxGlassMaterial, VoxGroup, VoxReader, VoxShape, VoxTransform


class Vox2UsdConverter:
    """Entry point and main logic for translating vox file to usd.

    Attributes:
        vox_file_path: The path to the vox file to convert.
        voxel_size: The size in meters for each voxel. Also determines the spacing between voxels.
        gamma_correct: Whether to gamma correct the colors from MagicaVoxel.
        gamma_value: The target gamma for voxel colors.
        use_physics: Whether to set physics attributes on point instances.
        flatten: Output a single USD file rather than utilizing payloads.
        total_voxels: A counter for the number of converted voxels.
        total_triangles: A counter for the number of converted triangles.
        used_mtls: A dictionary tracking the materials actually used from the vox palette and binding targets.
        asset_name: The name of the asset derived from the vox file name.
        output_dir: The output directory which is the same as the parent directory of the vox file.
        output_file_name: The name of the model stage layer.
        mesh_payload_identifier: The file path for the mesh payload.
        points_payload_identifier: The file path for the points payload.
    """

    def __init__(
        self,
        vox_file,
        voxel_size=DEFAULT_METERS_PER_VOXEL,
        gamma_correct=True,
        gamma_value=2.2,
        use_physics=False,
        flatten=False,
    ):
        """Vox2UsdConverter constructor

        Args:
            vox_file: The path to the vox file to convert.
            voxel_size: The size in meters for each voxel. Also determines the spacing between voxels.
            gamma_correct: Whether to gamma correct the colors from MagicaVoxel.
            gamma_value: The target gamma for voxel colors.
            use_physics: Whether to set physics attributes on point instances.
            flatten: Output a single USD file.
        """
        self.vox_file_path = vox_file
        self.voxel_size = voxel_size
        self.gamma_correct = gamma_correct
        self.gamma_value = gamma_value
        self.use_physics = use_physics
        self.flatten = flatten
        self.total_voxels = 0
        self.total_triangles = 0
        self.used_mtls = {}
        self.asset_name = os.path.splitext(os.path.basename(self.vox_file_path))[0]
        self.output_dir = os.path.dirname(self.vox_file_path)
        self.output_file_name = f"{self.asset_name}.usd"
        self.mesh_payload_identifier = os.path.join(self.output_dir, f"{self.asset_name}.mesh.usdc")
        self.points_payload_identifier = os.path.join(self.output_dir, f"{self.asset_name}.points.usdc")

        VoxBaseMaterial.initialize(self.gamma_correct, self.gamma_value)

    def create_material(self, name, vox_mtl):
        """Creates a UsdShade.Material based on the provided MagicaVoxel material.

        Args:
            name (str): Material prim name
            vox_mtl (VoxBaseMaterial): The source vox material

        Returns:
            UsdShade.Material: The created USD material
        """
        mtl = UsdShade.Material.Define(self.model_stage, self.looks_scope.GetPath().AppendPath(name))
        self.shader_varset.SetVariantSelection(ShaderVariantSetNames.PREVIEW)
        with self.shader_varset.GetVariantEditContext():
            prvw_shader = UsdShade.Shader.Define(self.model_stage, mtl.GetPath().AppendPath("prvw_shader"))
            prvw_shader = vox_mtl.populate_usd_preview_surface(prvw_shader)
            mtl.CreateSurfaceOutput().ConnectToSource(prvw_shader.ConnectableAPI(), "surface")
        self.shader_varset.SetVariantSelection(ShaderVariantSetNames.OMNIVERSE)
        with self.shader_varset.GetVariantEditContext():
            omni_shader = UsdShade.Shader.Define(self.model_stage, mtl.GetPath().AppendPath("omni_shader"))
            omni_shader = vox_mtl.populate_omni_shader(omni_shader)
            mtl.CreateSurfaceOutput().ConnectToSource(omni_shader.ConnectableAPI(), "surface")

        return mtl

    def convert(self):
        """Performs the conversion.

        Call this function after initializing the Vox2UsdConverter object run the conversion.
        """
        print(f"\nImporting voxel file {self.vox_file_path}\n")

        import time

        time_start = time.time()
        VoxReader(self.vox_file_path).read()

        print("Starting conversion...")
        stage_layer = Sdf.Layer.CreateNew(os.path.join(self.output_dir, self.output_file_name), args={"format": "usda"})
        self.model_stage = Usd.Stage.Open(stage_layer.identifier)
        UsdGeom.SetStageUpAxis(self.model_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.model_stage, 1.0)
        asset_geom = UsdGeom.Xform.Define(self.model_stage, "/" + self.asset_name)
        self.model_stage.SetDefaultPrim(asset_geom.GetPrim())
        model_api = Usd.ModelAPI(asset_geom)
        model_api.SetKind(Kind.Tokens.component)
        model_api.SetAssetName(self.asset_name)
        model_api.SetAssetIdentifier(self.output_file_name)
        self.geometry_scope = UsdGeom.Scope.Define(
            self.model_stage, asset_geom.GetPath().AppendPath(GEOMETRY_SCOPE_NAME)
        )
        self.looks_scope = UsdGeom.Scope.Define(self.model_stage, asset_geom.GetPath().AppendPath(LOOKS_SCOPE_NAME))

        self.geo_varset = asset_geom.GetPrim().GetVariantSets().AddVariantSet("Geometry")
        for variant_name in GeometryVariantSetNames.values():
            self.geo_varset.AddVariant(variant_name)

        self.geo_varset.SetVariantSelection(GeometryVariantSetNames.POINT_INSTANCES)
        with self.geo_varset.GetVariantEditContext():
            self.point_shape_varset = asset_geom.GetPrim().GetVariantSets().AddVariantSet("PointShape")
            for variant_name in PointShapeVariantSetNames.values():
                self.point_shape_varset.AddVariant(variant_name)
        self.shader_varset = asset_geom.GetPrim().GetVariantSets().AddVariantSet("Shader")
        for variant_name in ShaderVariantSetNames.values():
            self.shader_varset.AddVariant(variant_name)

        GreedyMeshing.generate(self.voxel_size)

        for index in VoxBaseMaterial.used_palette_ids:
            mtl = VoxBaseMaterial.get(index)
            mtl = self.create_material(f"VoxelMtl_{mtl.get_display_id()}", mtl)
            self.used_mtls[index] = {
                "mtl": mtl,
                GeometryVariantSetNames.MERGED_MESHES: [],
                GeometryVariantSetNames.POINT_INSTANCES: {
                    PointShapeVariantSetNames.CUBES: [],
                    PointShapeVariantSetNames.SPHERES: [],
                    PointShapeVariantSetNames.STUDS: [],
                },
            }

        mesh_stage = Usd.Stage.CreateNew(self.mesh_payload_identifier)
        points_stage = Usd.Stage.CreateNew(self.points_payload_identifier)

        root_xform_node = VoxTransform.get_top_nodes()[0]
        root_group_node = root_xform_node.get_child()

        self._convert_node(mesh_stage, root_xform_node, mesh_stage.GetPseudoRoot(), self._voxels2mesh)
        self._convert_node(points_stage, root_xform_node, points_stage.GetPseudoRoot(), self._voxels2point_instances)

        default_prim = mesh_stage.GetPrimAtPath(
            mesh_stage.GetPseudoRoot().GetPath().AppendPath(f"VoxelGroup_{root_group_node.node_id}")
        )
        mesh_stage.SetDefaultPrim(default_prim)
        default_prim = points_stage.GetPrimAtPath(
            points_stage.GetPseudoRoot().GetPath().AppendPath(f"VoxelGroup_{root_group_node.node_id}")
        )
        points_stage.SetDefaultPrim(default_prim)
        mesh_stage.Save()
        points_stage.Save()

        voxel_root_path = self.geometry_scope.GetPath().AppendChild("VoxelRoot")
        self.voxel_root = self.model_stage.OverridePrim(voxel_root_path)

        self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
        with self.geo_varset.GetVariantEditContext():
            self.voxel_root.GetPayloads().AddPayload(Sdf.Payload(f"./{self.asset_name}.mesh.usdc"))
        self.geo_varset.SetVariantSelection(GeometryVariantSetNames.POINT_INSTANCES)
        with self.geo_varset.GetVariantEditContext():
            self.voxel_root.GetPayloads().AddPayload(Sdf.Payload(f"./{self.asset_name}.points.usdc"))

        for mtl_id, item in self.used_mtls.items():
            mtl = item["mtl"]
            merged_mesh_targets = item[GeometryVariantSetNames.MERGED_MESHES]
            proto_targets = item[GeometryVariantSetNames.POINT_INSTANCES]

            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
            with self.geo_varset.GetVariantEditContext():
                for target_path in merged_mesh_targets:
                    target_prim = self._get_target_prim(target_path)
                    UsdShade.MaterialBindingAPI(target_prim).Bind(mtl)
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.POINT_INSTANCES)
            with self.geo_varset.GetVariantEditContext():
                for pt_shape_variant in PointShapeVariantSetNames.values():
                    self._bind_mtl_to_prototypes(mtl, proto_targets, pt_shape_variant)

        if self.flatten:
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
            self.shader_varset.SetVariantSelection(ShaderVariantSetNames.PREVIEW)
            self.model_stage.Export(os.path.join(self.output_dir, self.output_file_name), args={"format": "usdc"})
            del self.model_stage
            del mesh_stage
            del points_stage
            os.remove(self.mesh_payload_identifier)
            os.remove(self.points_payload_identifier)
        else:
            self.geo_varset.SetVariantSelection(GeometryVariantSetNames.MERGED_MESHES)
            self.shader_varset.SetVariantSelection(ShaderVariantSetNames.OMNIVERSE)
            self.point_shape_varset.SetVariantSelection(PointShapeVariantSetNames.CUBES)
            self.model_stage.Save()

        print(f"Converted {self.total_voxels} total voxels")
        print(f"Converted {self.total_triangles} total triangles")
        print(f"\nSuccessfully converted {self.vox_file_path} in {time.time() - time_start:.3f} sec")

    def _get_target_prim(self, target_path):
        """Converts a prim path from a payload stage to the model stage.

        After the payloads are added to the model stage, we need to convert the
        prim paths from the payload stages to the model stage for material binding.

        Args:
            target_path (Sdf.Path): The prim path to convert.

        Returns:
            Sdf.Path: The recomposed path for the target prim on the model stage.
        """
        target_path = target_path.pathString.split("/")
        target_path = "/".join(target_path[2:])
        target_path = self.voxel_root.GetPath().AppendPath(target_path)
        return self.model_stage.GetPrimAtPath(target_path)

    def _bind_mtl_to_prototypes(self, mtl, proto_targets, pt_shape_variant):
        """Binds the given material to the point instance prototypes.

        Args:
            pt_shape_variant: The shape variant to set before binding the material.
            proto_targets: The prototypes to bind to.
            mtl (UsdShade.Material): The material to bind.
        """
        self.point_shape_varset.SetVariantSelection(pt_shape_variant)
        with self.point_shape_varset.GetVariantEditContext():
            self.voxel_root.GetVariantSet("PointShape").SetVariantSelection(pt_shape_variant)
            for target_path in proto_targets[pt_shape_variant]:
                target_prim = self._get_target_prim(target_path)
                UsdShade.MaterialBindingAPI(target_prim).Bind(mtl)

    def _convert_node(self, stage, node, parent_prim, voxel_convert_func):
        """Converts a vox node to USD.

        Recursively traverses the vox data to convert all of the vox nodes. The vox transform
        nodes apply their data to their child group and shape nodes in order to reduce prim count.
        This function takes a stage in order to be reused for both mesh and points payloads.

        Args:
            stage (Usd.Stage): The payload stage to populate the data in.
            node: The node to convert.
            parent_prim: The prim to treat as the parent of the current node.
            voxel_convert_func: The function used to convert VoxShape nodes.
        """
        if isinstance(node, VoxTransform):
            xform_child = node.get_child()
            if isinstance(xform_child, VoxGroup):
                xform_path = parent_prim.GetPath().AppendPath(f"VoxelGroup_{xform_child.node_id}")
                xform = UsdGeom.Xform.Define(stage, xform_path)
                xform.AddTransformOp().Set(Gf.Matrix4d(*node.transform))
                for child in xform_child.children:
                    self._convert_node(stage, child, xform, voxel_convert_func)
            elif isinstance(xform_child, VoxShape):
                voxel_convert_func(stage, node, xform_child, parent_prim)
        else:
            raise RuntimeError(f"Expected VoxTransform node. Got {node.__class__}.")

    @staticmethod
    def _set_pivot(vox_model, xformable):
        """Modify the transform to apply the pivot at the bottom of the shape for easier placement."""
        # TODO: This doesn't actually work for rotated models. The pivot doesn't end up at the bottom
        # MV pivot is at the center. I prefer to have it at the bottom.
        # I'm tweaking the translation here to counteract that.
        xform_attr = xformable.GetPrim().GetAttribute("xformOp:transform")
        curr_xform = xform_attr.Get()
        # Need to figure out the local
        up_vector = curr_xform * Gf.Vec4d(0, 0, 1, 1)
        up_vector = Gf.Vec3d(up_vector[0:3]).GetNormalized()
        # copy bottom row of matrix since index operator is read only on Matrix4d
        trans_row = curr_xform[3]
        for index in range(3):
            if up_vector[index] != 0:
                trans_row[index] = trans_row[index] - int(vox_model.size[2] / 2.0)
                break

        curr_xform.SetRow(3, Gf.Vec4d(*trans_row))
        xform_attr.Set(curr_xform)

    def _add_binding_target(self, mtl_id, target_path, geometry_var, point_shape_var=None):
        """Stores target and material binding information

        Target and material binding information is stored in self.used_mtls to
        later perform the binding in the model stage.

        Args:
            mtl_id: The unique id for the material to use for binding.
            target_path: The prim path for that target that should be bound.
            geometry_var: The geometry variant associated with the target
            point_shape_var: The point shape variant associated with the target.
        """
        if geometry_var == GeometryVariantSetNames.MERGED_MESHES:
            self.used_mtls[mtl_id][geometry_var].append(target_path)
        elif geometry_var == GeometryVariantSetNames.POINT_INSTANCES:
            if point_shape_var is None:
                raise TypeError("You must provide a PointShapeVariantSetNames value for point_shape_var.")
            else:
                self.used_mtls[mtl_id][geometry_var][point_shape_var].append(target_path)

    def _set_mesh_attrs(self, shape_node, mesh):
        """Sets mesh prim attributes

        All defines geom subsets if the shape_node utilizes more than one material.

        Args:
            shape_node (VoxShape): Contains the source data to populate the mesh prim.
            mesh: The mesh prim to populate.
        """
        vertices = []
        total_face_count = 0
        display_colors = []
        opacity = []
        subsets = []
        for mtl_id, mesh_verts in shape_node.model.meshes.items():
            start_idx = len(vertices)
            vertices.extend(mesh_verts)
            end_idx = len(vertices)
            start_face_idx = int(start_idx / 4)
            end_face_idx = int(end_idx / 4)
            face_count = int(len(mesh_verts) / 4.0)
            total_face_count += face_count
            vox_mtl = VoxBaseMaterial.get(mtl_id)
            display_colors.extend([Gf.Vec3f(vox_mtl.color[0:3])] * face_count)
            if isinstance(vox_mtl, VoxGlassMaterial):
                opacity.extend([vox_mtl.get_opacity()] * face_count)
            else:
                opacity.extend([1.0] * face_count)
            subsets.append({"mtl_id": mtl_id, "start_face_idx": start_face_idx, "end_face_idx": end_face_idx})

        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([4] * total_face_count)
        mesh.CreateFaceVertexIndicesAttr(list(range(len(vertices))))
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(display_colors)
        mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.uniform).Set(opacity)

        if len(subsets) > 1:
            # Create geom subsets if the mesh uses more than one material.
            mesh_binding_api = UsdShade.MaterialBindingAPI(mesh.GetPrim())
            for item in subsets:
                subset = mesh_binding_api.CreateMaterialBindSubset(
                    "VoxelPart_{}".format(item["mtl_id"]),
                    Vt.IntArray(list(range(item["start_face_idx"], item["end_face_idx"]))),
                )
                self._add_binding_target(item["mtl_id"], subset.GetPath(), GeometryVariantSetNames.MERGED_MESHES)
        else:
            # This mesh uses only one material, so no need to create geom subsets.
            self._add_binding_target(subsets[0]["mtl_id"], mesh.GetPath(), GeometryVariantSetNames.MERGED_MESHES)

        self.total_triangles += total_face_count * 2

    def _voxels2mesh(self, stage, xform_node, shape_node, parent_prim):
        """Converts a VoxShape into a UsdGeom.Mesh

        Args:
            stage (Usd.Stage): The payload stage to operate on.
            xform_node (VoxTransform): The transform data for this shape.
            shape_node (VoxShape): The node containing the source data to convert.
            parent_prim: The parent prim for the mesh.
        """
        self.total_voxels += len(shape_node.model.voxels)
        mesh = UsdGeom.Mesh.Define(stage, parent_prim.GetPath().AppendPath(f"VoxelModel_{shape_node.node_id}"))
        mesh.AddTransformOp().Set(Gf.Matrix4d(*xform_node.transform))
        # self._set_pivot(shape_node.model, mesh)
        self._set_mesh_attrs(shape_node, mesh)

    def _voxels2point_instances(self, stage, xform_node, shape_node, parent_prim):
        """Converts a VoxShape into a UsdGeom.PointInstancer

        Creates prototypes for different variant point shapes based on PointShapeVariantSetNames.

        Args:
            stage (Usd.Stage): The payload stage to operate on.
            xform_node (VoxTransform): The transform data for this shape.
            shape_node (VoxShape): The node containing the source data to convert.
            parent_prim: The parent prim for the mesh.
        """
        instancer = UsdGeom.PointInstancer.Define(
            stage, parent_prim.GetPath().AppendPath(f"VoxelModel_{shape_node.node_id}")
        )
        instancer.AddTransformOp().Set(Gf.Matrix4d(*xform_node.transform))
        # self._set_pivot(shape_node.model, instancer)
        instancer.CreatePrototypesRel()
        proto_container = stage.OverridePrim(instancer.GetPath().AppendPath("Prototypes"))
        mtl2proto_id = {}
        default_prim = stage.GetPseudoRoot().GetChildren()[0]
        if not default_prim.HasVariantSets():
            pt_shape_varset = default_prim.GetVariantSets().AddVariantSet("PointShape")
            for variant_name in PointShapeVariantSetNames.values():
                pt_shape_varset.AddVariant(variant_name)
        else:
            pt_shape_varset = default_prim.GetVariantSet("PointShape")

        ids = []
        positions = []
        for voxel in shape_node.model.voxels:
            mtl_id = voxel[3]
            if mtl_id not in mtl2proto_id:
                mtl2proto_id[mtl_id] = len(proto_container.GetChildren())
                pt_shape_varset.SetVariantSelection(PointShapeVariantSetNames.CUBES)
                with pt_shape_varset.GetVariantEditContext():
                    self._create_cube_geom(stage, instancer, proto_container, mtl_id)
                pt_shape_varset.SetVariantSelection(PointShapeVariantSetNames.SPHERES)
                with pt_shape_varset.GetVariantEditContext():
                    self._create_sphere_geom(stage, instancer, proto_container, mtl_id)
                pt_shape_varset.SetVariantSelection(PointShapeVariantSetNames.STUDS)
                with pt_shape_varset.GetVariantEditContext():
                    self._create_stud_geom(stage, instancer, proto_container, mtl_id)

            ids.append(mtl2proto_id[mtl_id])
            position = [float(coord) * self.voxel_size for coord in voxel[:3]]
            # XY center the local origin on the model
            position = [
                position[0] - int(shape_node.model.size[0] / 2.0),
                position[1] - int(shape_node.model.size[1] / 2.0),
                position[2] - int(shape_node.model.size[2] / 2.0),
            ]
            positions.append(position)
        instancer.CreateProtoIndicesAttr()
        instancer.CreatePositionsAttr()
        instancer.GetProtoIndicesAttr().Set(ids)
        instancer.GetPositionsAttr().Set(positions)

    def _create_cube_geom(self, stage, instancer, proto_container, mtl_id):
        """Create a cube shape prototype.

        Creates a prototype and sets the relationship to its instancer.

        Args:
            stage (Usd.Stage): The payload stage to operate on.
            instancer (UsdGeom.PointInstancer): The instancer the prototype belongs to.
            proto_container: The parent prim containing prototypes.
            mtl_id: The unique id for the material for this prototype.
        """
        mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
        cube = UsdGeom.Cube.Define(stage, proto_container.GetPath().AppendPath(f"VoxelCube_{mtl_display_id}"))
        cube.CreateSizeAttr(self.voxel_size)
        self._add_binding_target(
            mtl_id,
            cube.GetPath(),
            GeometryVariantSetNames.POINT_INSTANCES,
            point_shape_var=PointShapeVariantSetNames.CUBES,
        )
        self._set_common_point_attrs(cube, instancer, mtl_id)

    def _create_sphere_geom(self, stage, instancer, proto_container, mtl_id):
        """Create a sphere shape prototype.

        Creates a prototype and sets the relationship to its instancer.

        Args:
            stage (Usd.Stage): The payload stage to operate on.
            instancer (UsdGeom.PointInstancer): The instancer the prototype belongs to.
            proto_container: The parent prim containing prototypes.
            mtl_id: The unique id for the material for this prototype.
        """
        mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
        sphere_path = proto_container.GetPath().AppendPath(f"VoxelSphere_{mtl_display_id}")
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.CreateRadiusAttr(self.voxel_size / 2.0)
        self._add_binding_target(
            mtl_id,
            sphere.GetPath(),
            GeometryVariantSetNames.POINT_INSTANCES,
            point_shape_var=PointShapeVariantSetNames.SPHERES,
        )
        self._set_common_point_attrs(sphere, instancer, mtl_id)

    def _create_stud_geom(self, stage, instancer, proto_container, mtl_id):
        """Create a stud shape prototype.

        Creates a prototype and sets the relationship to its instancer.

        Args:
            stage (Usd.Stage): The payload stage to operate on.
            instancer (UsdGeom.PointInstancer): The instancer the prototype belongs to.
            proto_container: The parent prim containing prototypes.
            mtl_id: The unique id for the material for this prototype.
        """
        mtl_display_id = VoxBaseMaterial.get(mtl_id).get_display_id()
        mesh = UsdGeom.Mesh.Define(stage, proto_container.GetPath().AppendPath(f"VoxelStud_{mtl_display_id}"))
        mesh.CreatePointsAttr(StudMesh.points)
        mesh.CreateFaceVertexCountsAttr(StudMesh.face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(StudMesh.face_vertex_indices)
        mesh.CreateNormalsAttr(StudMesh.normals)
        mesh.SetNormalsInterpolation(StudMesh.normals_interpolation)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self._add_binding_target(
            mtl_id,
            mesh.GetPath(),
            GeometryVariantSetNames.POINT_INSTANCES,
            point_shape_var=PointShapeVariantSetNames.STUDS,
        )
        self._set_common_point_attrs(mesh, instancer, mtl_id)

    def _set_common_point_attrs(self, geom, instancer, mtl_id):
        """Sets common point prototype attributes.

        Args:
            geom: The prototype prim to set attributes for.
            instancer: The prototype's instancer.
            mtl_id: The unique id for the material for this prototype.
        """
        # Move voxel pivot to the front-bottom-left like MV voxels
        geom.AddTranslateOp().Set((0.5, 0.5, 0.5))

        vox_mtl = VoxBaseMaterial.get(mtl_id)
        geom.CreateDisplayColorAttr([vox_mtl.color[0:3]])
        if isinstance(vox_mtl, VoxGlassMaterial):
            geom.CreateDisplayOpacityAttr([vox_mtl.get_opacity()])
        if self.use_physics:
            physics_apis = Sdf.TokenListOp.Create(["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"])
            shape_prim = geom.GetPrim()
            shape_prim.SetMetadata("apiSchemas", physics_apis)
            shape_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(1)
            shape_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(0)
            shape_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(1)
            shape_prim.CreateAttribute("physics:startsAsleep", Sdf.ValueTypeNames.Bool).Set(1)

        instancer.GetPrototypesRel().AddTarget(geom.GetPath())
