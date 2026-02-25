import bpy
import math
import mathutils
import json
import os
import sys
import numpy as np
from contextlib import contextmanager

# ------------------------------------------------------------------------
# OUTPUT SUPPRESSION
# ------------------------------------------------------------------------

@contextmanager
def suppress_blender_output():
    """Suppress Blender's render output (Saved: ..., Time: ...)."""
    import io
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# ------------------------------------------------------------------------
# 1. SCENE & RENDERING SETUP
# ------------------------------------------------------------------------

def setup_scene():
    """Initializes a high-visibility rendering environment."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    if hasattr(bpy.context.scene.view_settings, 'view_transform'):
        bpy.context.scene.view_settings.view_transform = 'Standard'
    
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.eevee.use_soft_shadows = False
    
    # Disable film transparency - render with world background
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

def setup_black_background():
    """Sets up pure black background for body rendering."""
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("AutoRig_Studio")
    
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    
    # Pure black background
    bg = nodes.new(type='ShaderNodeBackground')
    bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs[1].default_value = 1.0
    out = nodes.new(type='ShaderNodeOutputWorld')
    world.node_tree.links.new(bg.outputs[0], out.inputs[0])

def setup_hdri_background(visible=True):
    """Sets up forest HDRI background with optional visibility.
    
    Args:
        visible: If True, HDRI is visible in render. If False, uses black background but HDRI still provides lighting.
    """
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("AutoRig_Studio")
    
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    
    # Setup Environment Texture for HDRI
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-800, 300)
    
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-600, 300)
    
    env_tex = nodes.new(type='ShaderNodeTexEnvironment')
    env_tex.location = (-400, 300)
    
    # Try to find forest.exr in Blender's datafiles
    import os
    blender_path = os.path.dirname(bpy.app.binary_path)
    possible_paths = [
        os.path.join(blender_path, "4.2", "datafiles", "studiolights", "world", "forest.exr"),
        os.path.join(blender_path, "datafiles", "studiolights", "world", "forest.exr"),
        os.path.join(blender_path, "..", "share", "blender", "4.2", "datafiles", "studiolights", "world", "forest.exr"),
    ]
    
    hdri_path = None
    for path in possible_paths:
        if os.path.exists(path):
            hdri_path = path
            print(f"### [AutoRig] Found forest.exr at: {path}", flush=True)
            break
    
    if hdri_path:
        try:
            env_tex.image = bpy.data.images.load(hdri_path)
        except:
            print(f"### [AutoRig] Failed to load forest.exr, using neutral gray", flush=True)
    else:
        print(f"### [AutoRig] forest.exr not found, using neutral gray background", flush=True)
    
    bg_hdri = nodes.new(type='ShaderNodeBackground')
    bg_hdri.location = (-200, 400)
    bg_hdri.inputs[1].default_value = 1.0  # Strength
    
    if not visible:
        # Use Light Path node to only apply HDRI to lighting, not camera rays
        light_path = nodes.new(type='ShaderNodeLightPath')
        light_path.location = (-400, 100)
        
        # Black background for camera
        bg_black = nodes.new(type='ShaderNodeBackground')
        bg_black.location = (-200, 200)
        bg_black.inputs[0].default_value = (0, 0, 0, 1)  # Black
        
        # Mix shader - use HDRI for lighting, black for camera
        mix = nodes.new(type='ShaderNodeMixShader')
        mix.location = (0, 300)
        
        out = nodes.new(type='ShaderNodeOutputWorld')
        out.location = (200, 300)
        
        # Connect nodes
        world.node_tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        world.node_tree.links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
        world.node_tree.links.new(env_tex.outputs['Color'], bg_hdri.inputs['Color'])
        world.node_tree.links.new(light_path.outputs['Is Camera Ray'], mix.inputs[0])  # Factor
        world.node_tree.links.new(bg_hdri.outputs['Background'], mix.inputs[1])  # HDRI for non-camera rays
        world.node_tree.links.new(bg_black.outputs['Background'], mix.inputs[2])  # Black for camera rays
        world.node_tree.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    else:
        # Simple setup - HDRI visible everywhere
        out = nodes.new(type='ShaderNodeOutputWorld')
        out.location = (0, 300)
        
        # Connect nodes
        world.node_tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        world.node_tree.links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
        world.node_tree.links.new(env_tex.outputs['Color'], bg_hdri.inputs['Color'])
        world.node_tree.links.new(bg_hdri.outputs['Background'], out.inputs['Surface'])
    
    if hasattr(bpy.context.scene.view_settings, 'view_transform'):
        bpy.context.scene.view_settings.view_transform = 'Standard'
    
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.eevee.use_soft_shadows = False
    
    # Disable film transparency - render with world background
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

def setup_lighting_camera_relative(cam, center, size):
    """Sets up lighting relative to camera position for uniform illumination."""
    # Clear existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Calculate camera direction
    cam_to_center = (center - cam.location).normalized()
    
    # Key light: slightly offset from camera (upper right)
    key_offset = mathutils.Vector((size * 0.5, size * 0.3, size * 0.8))
    bpy.ops.object.light_add(type='AREA', location=cam.location + key_offset)
    key = bpy.context.object
    key.data.energy = 40.0 * size  # Reduced from 80.0
    key.data.size = size * 1.5
    direction = (center - key.location).to_track_quat('-Z', 'Y')
    key.rotation_euler = direction.to_euler()
    
    # Fill light: opposite side from key
    fill_offset = mathutils.Vector((-size * 0.5, -size * 0.3, size * 0.5))
    bpy.ops.object.light_add(type='AREA', location=cam.location + fill_offset)
    fill = bpy.context.object
    fill.data.energy = 25.0 * size  # Reduced from 50.0
    fill.data.size = size * 2.0
    direction = (fill.location - center).to_track_quat('Z', 'Y')
    fill.rotation_euler = direction.to_euler()
    
    # Soft ambient from camera direction
    bpy.ops.object.light_add(type='AREA', location=cam.location + mathutils.Vector((0, 0, size * 0.2)))
    ambient = bpy.context.object
    ambient.data.energy = 15.0 * size  # Reduced from 30.0
    ambient.data.size = size * 3.0
    direction = (center - ambient.location).to_track_quat('-Z', 'Y')
    ambient.rotation_euler = direction.to_euler()

def setup_lighting(center, size):
    """Sets up balanced Studio Lighting for natural depth cues."""
    # Key light - main light source from upper right
    bpy.ops.object.light_add(type='AREA', location=center + mathutils.Vector((size, -size, size*2)))
    key = bpy.context.object
    key.data.energy = 120.0 * size
    key.data.size = size * 2.0
    key.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    # Fill light - softer light from opposite side to reduce harsh shadows
    bpy.ops.object.light_add(type='AREA', location=center + mathutils.Vector((-size*2, -size, size)))
    fill = bpy.context.object
    fill.data.energy = 70.0 * size
    fill.data.size = size * 3.0
    fill.rotation_euler = (math.radians(60), 0, math.radians(-45))
    
    # Back light - stronger rim light from behind to separate subject from background
    bpy.ops.object.light_add(type='AREA', location=center + mathutils.Vector((0, size*2, size*1.5)))
    back = bpy.context.object
    back.data.energy = 150.0 * size  # Stronger back light
    back.data.size = size * 2.5
    back.rotation_euler = (math.radians(-60), 0, 0)
    
    # Soft bounce light from below to lift shadows naturally
    bpy.ops.object.light_add(type='AREA', location=center + mathutils.Vector((0, 0, -size*1.2)))
    bounce = bpy.context.object
    bounce.data.energy = 35.0 * size
    bounce.data.size = size * 4.0
    bounce.rotation_euler = (math.radians(0), 0, 0)

def prepare_materials(meshes, force_matte=False):
    """Applies a skin tone material if model has no textures, otherwise preserves materials."""
    # Create skin tone material (same as used for hand detection)
    mat_name = "AutoRig_SkinTone"
    skin_mat = bpy.data.materials.get(mat_name)
    if not skin_mat:
        skin_mat = bpy.data.materials.new(name=mat_name)
        skin_mat.use_nodes = True
        bsdf = skin_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Natural skin tone color (peachy beige) - same as hand detection material
            bsdf.inputs['Base Color'].default_value = (0.95, 0.76, 0.65, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.5
            bsdf.inputs['Specular IOR Level'].default_value = 0.3
            bsdf.inputs['Subsurface Weight'].default_value = 0.05  # Slight SSS for realism
            bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.5, 0.25)

    for obj in meshes:
        if force_matte:
            # Force apply skin tone (for consistency)
            obj.data.materials.clear()
            obj.data.materials.append(skin_mat)
        else:
            # Check if mesh has valid textured materials
            has_valid_material = False
            if obj.data.materials:
                for slot in obj.material_slots:
                    if slot.material and slot.material.use_nodes:
                        # Check if material has any texture nodes (Image Texture nodes)
                        for node in slot.material.node_tree.nodes:
                            if node.type == 'TEX_IMAGE' and node.image:
                                has_valid_material = True
                                break
                    if has_valid_material:
                        break
            
            if not has_valid_material:
                # No valid textures found - apply skin tone
                obj.data.materials.clear()
                obj.data.materials.append(skin_mat)
            else:
                # Keep existing materials, just adjust metallic to prevent reflections
                for slot in obj.material_slots:
                    if slot.material and slot.material.use_nodes:
                        bsdf = slot.material.node_tree.nodes.get("Principled BSDF")
                        if bsdf:
                            bsdf.inputs['Metallic'].default_value = 0.0

def apply_temp_skin_material(mesh):
    """
    Apply temporary skin tone material for better hand detection.
    Always applies skin tone - improves detection on all models.
    Returns dict with original materials for restoration.
    """
    # Store original materials
    original_mats = {}
    for i, slot in enumerate(mesh.material_slots):
        original_mats[i] = slot.material
    
    print(f"### [AutoRig] Applying temporary skin tone for optimal hand detection")
    
    # Create skin tone material
    mat_name = "AutoRig_TempSkin"
    skin_mat = bpy.data.materials.get(mat_name)
    if not skin_mat:
        skin_mat = bpy.data.materials.new(name=mat_name)
        skin_mat.use_nodes = True
        bsdf = skin_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Natural skin tone color (peachy beige)
            bsdf.inputs['Base Color'].default_value = (0.95, 0.76, 0.65, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.5
            bsdf.inputs['Specular IOR Level'].default_value = 0.3
            bsdf.inputs['Subsurface Weight'].default_value = 0.05  # Slight SSS for realism
            bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.5, 0.25)
    
    # Apply to all material slots
    mesh.data.materials.clear()
    mesh.data.materials.append(skin_mat)
    
    return original_mats

def restore_original_materials(mesh, original_mats):
    """Restore original materials after temporary skin material."""
    print(f"### [AutoRig] Restoring original materials")
    mesh.data.materials.clear()
    for i in sorted(original_mats.keys()):
        if original_mats[i]:
            mesh.data.materials.append(original_mats[i])
        else:
            mesh.data.materials.append(None)

def frame_camera(objs, camera, padding=1.55):
    """Perspective framing. Sets sensor and lens to prevent clipping head/feet."""
    bpy.context.view_layer.update()
    bbox = []
    for obj in objs:
        if obj.type == 'MESH':
            bbox.extend([obj.matrix_world @ mathutils.Vector(v) for v in obj.bound_box])
    if not bbox: return mathutils.Vector((0,0,0)), 5.0, 1.0
    
    min_c = [min(v[i] for v in bbox) for i in range(3)]
    max_c = [max(v[i] for v in bbox) for i in range(3)]
    center = mathutils.Vector([(min_c[i] + max_c[i]) / 2 for i in range(3)])
    
    w, d, h = max_c[0]-min_c[0], max_c[1]-min_c[1], max_c[2]-min_c[2]
    size = max(w, d, h)
    
    camera.data.type = 'PERSP'
    camera.data.lens = 50.0 
    camera.data.sensor_fit = 'AUTO'
    
    fov = camera.data.angle
    dist = (size / 2) / math.tan(fov / 2) * padding
    dist = max(dist, 0.1)
    
    camera.data.clip_start = dist * 0.01
    camera.data.clip_end = dist * 20.0
    return center, dist, size

# ------------------------------------------------------------------------
# 2. MESH OPTIMIZATION
# ------------------------------------------------------------------------

def basic_cleanup(mesh_obj):
    """Standard non-destructive cleanup."""
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.object.mode_set(mode='OBJECT')

def recover_broken_mesh(mesh_obj):
    """Aggressive but fast recovery for AI meshes that fail rigging."""
    print("### [AutoRig] Starting Fast Recovery...")
    bpy.context.view_layer.objects.active = mesh_obj
    
    v_count = len(mesh_obj.data.vertices)
    if v_count > 80000:
        print(f"### [AutoRig] Step 1: Reducing vertex density ({v_count} -> 80k)...")
        dec = mesh_obj.modifiers.new(name="AutoRig_FastDec", type='DECIMATE')
        dec.ratio = 80000 / v_count
        bpy.ops.object.modifier_apply(modifier="AutoRig_FastDec")
    
    bpy.ops.object.mode_set(mode='EDIT')
    print("### [AutoRig] Step 2: Welding vertices...")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.005)
    print("### [AutoRig] Step 3: Deleting floating parts...")
    bpy.ops.mesh.delete_loose()
    print("### [AutoRig] Step 4: Closing small mesh gaps...")
    bpy.ops.mesh.fill_holes(sides=4)
    bpy.ops.object.mode_set(mode='OBJECT')

def fix_bone_orientations(arma):
    """
    Set proper bone rolls for natural articulation.
    Based on Rigify standards for humanoid rigs.
    """
    bpy.context.view_layer.objects.active = arma
    bpy.ops.object.mode_set(mode='EDIT')
    
    import mathutils
    
    for bone in arma.data.edit_bones:
        bone_name = bone.name.lower()
        
        # Get bone direction vector
        bone_vec = (bone.tail - bone.head).normalized()
        
        # Spine bones: Z-up, forward is Y
        if any(x in bone_name for x in ['spine', 'neck', 'head', 'root']):
            # Align to world Z (up)
            bone.align_roll(mathutils.Vector((0, 0, 1)))
        
        # Arms: point forward (Y forward in Blender bone space)
        elif any(x in bone_name for x in ['shoulder', 'arm', 'forearm', 'hand']):
            # For arms, we want elbows to bend forward (toward +Y in world)
            # Calculate a forward vector perpendicular to bone direction
            
            # Determine side (left vs right)
            is_left = '.l' in bone_name or 'left' in bone_name
            side_sign = 1.0 if is_left else -1.0
            
            # Forward direction for arms (toward front of body)
            forward = mathutils.Vector((0, 1, 0))  # +Y is forward
            
            # Calculate roll to make bone's X-axis point in forward direction
            # This makes elbows bend forward naturally
            bone.align_roll(forward)
        
        # Legs: point forward as well (knees bend forward)
        elif any(x in bone_name for x in ['pelvis', 'thigh', 'shin']):
            # Legs bend forward too
            forward = mathutils.Vector((0, 1, 0))
            bone.align_roll(forward)
        
        # Feet: flat on ground, toes forward
        elif any(x in bone_name for x in ['foot', 'toe']):
            # Feet should be flat, toes point forward
            # Z should point up even though foot is angled
            bone.align_roll(mathutils.Vector((0, 0, 1)))
        
        # Fingers: align along finger direction
        elif any(x in bone_name for x in ['thumb', 'index', 'middle', 'ring', 'pinky']):
            # Fingers curl toward palm
            # Determine which hand
            is_left = '.l' in bone_name or 'left' in bone_name
            
            # Fingers should curl downward (toward -Z generally)
            # But also influenced by hand orientation
            down = mathutils.Vector((0, 0, -1))
            bone.align_roll(down)
            # print(f"### [AutoRig] {bone.name}: finger roll → down", flush=True)  # Too verbose
        
        else:
            # Default: align to Z-up
            bone.align_roll(mathutils.Vector((0, 0, 1)))
    

    bpy.ops.object.mode_set(mode='OBJECT')

def fit_fingers_to_mesh(arma, mesh_obj):
    """
    Fit finger bones to the actual mesh geometry by:
    1. Finding cylindrical finger regions in the mesh
    2. Moving bones to fit inside those regions
    3. Ensuring no two finger chains occupy the same finger
    4. Centering bones within the finger mesh
    """
    import bmesh
    from collections import defaultdict
    
    print("### [AutoRig] Fitting finger bones to mesh geometry...", flush=True)
    
    bpy.context.view_layer.objects.active = arma
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Get mesh data
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    
    # Transform mesh vertices to world space
    mesh_world_matrix = mesh_obj.matrix_world
    world_verts = [mesh_world_matrix @ v.co for v in bm.verts]
    
    # Get hand bones to find hand region
    for side_suffix in ['.L', '.R']:
        hand_bone_name = f"hand{side_suffix}"
        hand_bone = arma.data.edit_bones.get(hand_bone_name)
        
        if not hand_bone:
            print(f"### [AutoRig] Hand bone {hand_bone_name} not found, skipping", flush=True)
            continue
        
        # Get wrist position (hand bone tail)
        wrist_pos = hand_bone.tail.copy()
        print(f"### [AutoRig] Processing {side_suffix} hand at wrist: {wrist_pos}", flush=True)
        
        # Find finger bones for this hand
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        finger_chains = {}
        
        for finger in finger_names:
            chain = []
            for seg in ['01', '02', '03']:
                bone_name = f"{finger}.{seg}{side_suffix}"
                bone = arma.data.edit_bones.get(bone_name)
                if bone:
                    chain.append(bone)
            if chain:
                finger_chains[finger] = chain
        
        if not finger_chains:
            print(f"### [AutoRig] No finger bones found for {side_suffix}", flush=True)
            continue
        
        # Find vertices in the hand region (within ~20cm of wrist)
        hand_region_verts = []
        for i, v in enumerate(world_verts):
            dist_to_wrist = (v - wrist_pos).length
            if dist_to_wrist < 0.20:  # 20cm radius
                hand_region_verts.append((i, v))
        
        print(f"### [AutoRig] Found {len(hand_region_verts)} vertices in hand region", flush=True)
        
        if len(hand_region_verts) < 50:
            print(f"### [AutoRig] Not enough vertices in hand region, skipping finger fitting", flush=True)
            continue
        
        # Get palm center (average of finger base bones)
        palm_bones = [finger_chains[f][0] for f in finger_chains if f != 'thumb' and finger_chains[f]]
        if palm_bones:
            palm_center = sum((b.head for b in palm_bones), mathutils.Vector()) / len(palm_bones)
        else:
            palm_center = wrist_pos
        
        # Direction from wrist to palm center (approximate hand direction)
        hand_forward = (palm_center - wrist_pos).normalized()
        
        # Find finger-like protrusions in mesh
        # Strategy: find vertices that extend beyond the palm in the hand-forward direction
        finger_verts = defaultdict(list)  # Will store vertices for each detected finger region
        
        # Sort hand vertices by their position along the hand-forward axis
        for idx, v in hand_region_verts:
            # Project vertex onto hand-forward direction
            forward_dist = (v - wrist_pos).dot(hand_forward)
            
            # Only consider vertices beyond the palm (forward_dist > some threshold)
            if forward_dist > 0.03:  # 3cm past wrist
                finger_verts['all'].append((idx, v, forward_dist))
        
        print(f"### [AutoRig] Found {len(finger_verts['all'])} finger region vertices", flush=True)
        
        if len(finger_verts['all']) < 20:
            continue
        
        # Cluster finger vertices into separate fingers based on X position (left-right spread)
        # Calculate perpendicular axis (across the hand)
        up_axis = mathutils.Vector((0, 0, 1))
        hand_right = hand_forward.cross(up_axis).normalized()
        if side_suffix == '.R':
            hand_right = -hand_right  # Flip for right hand
        
        # Group vertices by their position along the hand-right axis
        finger_clusters = defaultdict(list)
        for idx, v, forward_dist in finger_verts['all']:
            right_dist = (v - wrist_pos).dot(hand_right)
            # Quantize into ~5 regions
            cluster_id = int((right_dist + 0.06) / 0.02)  # ~2cm buckets
            finger_clusters[cluster_id].append((idx, v, forward_dist))
        
        # Sort clusters by position (thumb to pinky)
        sorted_clusters = sorted(finger_clusters.items(), key=lambda x: x[0])
        print(f"### [AutoRig] Detected {len(sorted_clusters)} finger clusters", flush=True)
        
        # Map bones to mesh clusters
        # Match expected finger order: thumb (most separated), index, middle, ring, pinky
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        # For each finger chain, find the best matching cluster
        assigned_clusters = set()
        
        for finger in finger_order:
            if finger not in finger_chains:
                continue
            
            chain = finger_chains[finger]
            if not chain:
                continue
            
            # Get current bone position
            bone_tip = chain[-1].tail  # Tip of last bone segment
            bone_base = chain[0].head  # Base of first bone segment
            
            # Find best matching cluster (closest to bone position)
            best_cluster = None
            best_dist = float('inf')
            
            for cluster_id, verts in sorted_clusters:
                if cluster_id in assigned_clusters:
                    continue
                
                # Get cluster center
                cluster_center = sum((v for _, v, _ in verts), mathutils.Vector()) / len(verts)
                
                # Distance from bone base to cluster
                dist = (cluster_center - bone_base).length
                
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cluster_id
            
            if best_cluster is None:
                print(f"### [AutoRig] No cluster found for {finger}{side_suffix}", flush=True)
                continue
            
            assigned_clusters.add(best_cluster)
            cluster_verts = finger_clusters[best_cluster]
            
            # Calculate the centerline of this finger cluster
            # Sort vertices by forward distance to get the finger axis
            cluster_verts.sort(key=lambda x: x[2])  # Sort by forward_dist
            
            # Split into segments (base, middle, tip)
            n = len(cluster_verts)
            if n < 6:
                continue
            
            seg_size = n // 3
            segments = [
                cluster_verts[:seg_size],
                cluster_verts[seg_size:2*seg_size],
                cluster_verts[2*seg_size:]
            ]
            
            # Calculate center of each segment
            segment_centers = []
            for seg in segments:
                if seg:
                    center = sum((v for _, v, _ in seg), mathutils.Vector()) / len(seg)
                    segment_centers.append(center)
            
            # Move bones to segment centers
            for i, bone in enumerate(chain):
                if i < len(segment_centers):
                    # Move bone head to segment center
                    old_head = bone.head.copy()
                    new_head = segment_centers[i]
                    
                    # Calculate offset
                    offset = new_head - old_head
                    
                    # Move head (and tail follows for connected bones)
                    bone.head = new_head
                    
                    # Set tail to next segment center or extend in same direction
                    if i + 1 < len(segment_centers):
                        bone.tail = segment_centers[i + 1]
                    elif i > 0:
                        # Extend in same direction as previous segment
                        direction = (new_head - segment_centers[i-1]).normalized()
                        bone.tail = new_head + direction * 0.02  # 2cm segment
                    
                    print(f"### [AutoRig] Moved {bone.name} to mesh center", flush=True)
        
        print(f"### [AutoRig] Finger fitting complete for {side_suffix} hand", flush=True)
    
    bm.free()
    bpy.ops.object.mode_set(mode='OBJECT')
    print("### [AutoRig] Finger mesh fitting complete", flush=True)

# ------------------------------------------------------------------------
# 3. MAIN TASK DISPATCHER
# ------------------------------------------------------------------------

def frame_hand_camera(skel, hand, cam, padding=1.2):
    wrist_j = f"Hand.{hand}"
    
    finger_joints = [
        f"Thumb.1.{hand}", f"Thumb.2.{hand}", f"Thumb.3.{hand}",
        f"Index.1.{hand}", f"Index.2.{hand}", f"Index.3.{hand}",
        f"Middle.1.{hand}", f"Middle.2.{hand}", f"Middle.3.{hand}",
        f"Ring.1.{hand}", f"Ring.2.{hand}", f"Ring.3.{hand}",
        f"Pinky.1.{hand}", f"Pinky.2.{hand}", f"Pinky.3.{hand}"
    ]
    
    existing_joints = [j for j in finger_joints + [wrist_j] if j in skel and np.linalg.norm(skel[j]) > 0.001]
    
    points = [mathutils.Vector(skel[j]) for j in existing_joints]
    if len(points) < 3:
        print(f"### [AutoRig] Hand {hand}: Not enough joints detected, skipping")
        return None, None, None, None

    center = sum(points, mathutils.Vector()) / len(points)
    
    # Calculate orientation
    if wrist_j not in skel:
        print(f"### [AutoRig] Hand {hand}: Missing wrist joint, skipping")
        return None, None, None, None
        
    wrist_pos = mathutils.Vector(skel[wrist_j])
    forearm_j = f"Forearm.{hand}"
    forearm_pos = mathutils.Vector(skel[forearm_j]) if forearm_j in skel else wrist_pos - mathutils.Vector((0,0,0.1))
    
    z_axis = (wrist_pos - forearm_pos).normalized() # Points up the arm
    
    # Get a point in the middle of the palm - check if joints exist
    palm_joints = [f"Index.1.{hand}", f"Middle.1.{hand}", f"Ring.1.{hand}"]
    palm_points = [mathutils.Vector(skel[j]) for j in palm_joints if j in skel]
    
    if len(palm_points) < 2:
        # Fallback: use center of all finger points
        palm_center = center
    else:
        palm_center = sum(palm_points, mathutils.Vector()) / len(palm_points)
    
    # Y-axis points from wrist to palm center
    y_axis = (palm_center - wrist_pos).normalized()
    
    # X-axis is the cross product
    x_axis = y_axis.cross(z_axis).normalized()
    
    # Recalculate y_axis to ensure orthogonality
    y_axis = z_axis.cross(x_axis).normalized()

    rot_matrix = mathutils.Matrix([x_axis, y_axis, z_axis]).transposed()
    
    # Find bounding box in local coordinates
    local_points = [rot_matrix.inverted() @ (p - center) for p in points]
    max_dims = [max(abs(p[i]) for p in local_points) for i in range(3)]
    
    size = max(max_dims) * 2.0
    
    cam.data.type = 'PERSP'
    cam.data.lens = 85.0 
    cam.data.sensor_fit = 'AUTO'
    fov = cam.data.angle
    
    dist = (size / 2) / math.tan(fov / 2) * padding
    dist = max(dist, 0.1)
    

    
    cam.data.clip_start = dist * 0.1
    cam.data.clip_end = dist * 2.0
    
    return center, dist, size, rot_matrix

# ------------------------------------------------------------------------
# 3. MAIN TASK DISPATCHER
# ------------------------------------------------------------------------

def run_task():
    try:
        idx = sys.argv.index("--")
        with open(sys.argv[idx + 1], 'r') as f: cfg = json.load(f)
        print(f"### [AutoRig] Mode: {cfg.get('mode', 'UNKNOWN')}", flush=True)
    except Exception as e:
        print(f"### [AutoRig] Failed to load config: {e}", flush=True)
        return

    setup_scene()
    
    try:
        ext = os.path.splitext(cfg['model_path'])[1].lower()
        if ext == '.obj': bpy.ops.wm.obj_import(filepath=cfg['model_path'])
        elif ext == '.fbx': bpy.ops.import_scene.fbx(filepath=cfg['model_path'], use_anim=False)
        elif ext in ['.glb', '.gltf']: bpy.ops.import_scene.gltf(filepath=cfg['model_path'])
    except: return

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    for o in list(bpy.data.objects):
        if o.type == 'ARMATURE': bpy.data.objects.remove(o, do_unlink=True)
    
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not meshes: return
    

    
    # Store original mesh origins before any processing
    original_origins = {}
    for mesh in meshes:
        original_origins[mesh.name] = mesh.location.copy()
    
    # Calculate the average original origin (in case of multiple meshes)
    if original_origins:
        avg_origin = mathutils.Vector((0, 0, 0))
        for origin in original_origins.values():
            avg_origin += origin
        avg_origin /= len(original_origins)

    
    # Don't join meshes - keep original geometry separated
    # Just use the first mesh for scaling reference
    bpy.ops.object.select_all(action='DESELECT')
    meshes[0].select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    main_mesh = meshes[0]
    
    # Apply transforms to all meshes WITHOUT changing their origins
    for mesh in meshes:
        mesh.select_set(True)
        mesh.location = (0,0,0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    if cfg['mode'] in ['preview', 'render_views']:
        # Keep original materials for all meshes
        prepare_materials(meshes, force_matte=False)
        
        bpy.ops.object.camera_add(); cam = bpy.context.object; bpy.context.scene.camera = cam
        center, dist, size = frame_camera(meshes, cam, padding=1.25)
        
        # Setup black background and studio lighting for body rendering
        setup_black_background()
        setup_lighting(center, size)
        
        res = cfg.get('res', 1024)
        bpy.context.scene.render.resolution_x = res
        bpy.context.scene.render.resolution_y = res

        if cfg['mode'] == 'preview':
            # Preview: front view of character (directly facing forward)
            cam.location = center + mathutils.Vector((0, -dist, 0))
            cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
            bpy.context.scene.render.filepath = cfg['output_path']
            with suppress_blender_output():
                bpy.ops.render.render(write_still=True)
        else:
            meta = []; os.makedirs(cfg['output_dir'], exist_ok=True)
            # 30 views for better triangulation:
            # - 20 views around horizontal ring at 18° increments (starting from front)
            # - 10 elevated views at 36° increments (45° elevation)
            for i in range(30):
                if i < 20:
                    # Horizontal ring: 20 views at 18° increments (360°/20)
                    # Start at 90° (front view) so first view faces character head-on
                    angle = math.radians(90 + i * 18)
                    cam.location = center + mathutils.Vector((math.cos(angle)*dist, math.sin(angle)*dist, 0))
                else:
                    # Elevated ring: 10 views at 36° increments (360°/10)
                    # Start at 90° (front) for consistency
                    angle = math.radians(90 + (i-20) * 36)
                    high_dist = dist * 0.707
                    z_offset = dist * 0.707
                    cam.location = center + mathutils.Vector((math.cos(angle)*high_dist, math.sin(angle)*high_dist, z_offset))
                
                cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
                
                p = os.path.join(cfg['output_dir'], f"v_{i}.png")
                bpy.context.scene.render.filepath = p
                with suppress_blender_output():
                    bpy.ops.render.render(write_still=True)
                P = cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get()) @ cam.matrix_world.inverted()
                meta.append([list(row) for row in P])
            with open(cfg['meta_path'], 'w') as f: json.dump(meta, f)

    elif cfg['mode'] == 'render_hand_views':
        print(f"### [AutoRig] Rendering hand views...", flush=True)
        try:
            # Setup HDRI background for natural lighting on hands
            setup_hdri_background()
            
            # Apply temporary skin tone material to all meshes
            original_materials = {}
            for i, mesh in enumerate(meshes):
                original_materials[i] = apply_temp_skin_material(mesh)
            
            bpy.ops.object.camera_add(); cam = bpy.context.object; bpy.context.scene.camera = cam
            
            res = cfg.get('res', 1024)
            bpy.context.scene.render.resolution_x = res
            bpy.context.scene.render.resolution_y = res
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.image_settings.color_mode = 'RGB'

            skel = cfg['skeleton']
            output_data = {}
            
            # First pass: calculate both hand distances to normalize framing
            hand_distances = {}
            for hand_suffix in ["L", "R"]:
                center, dist, size, rot_matrix = frame_hand_camera(skel, hand_suffix, cam)
                if center is not None:
                    hand_distances[hand_suffix] = {"center": center, "dist": dist, "size": size, "rot_matrix": rot_matrix}
            
            # Use the maximum distance for both hands to ensure consistent framing
            if hand_distances:
                max_dist = max(hd["dist"] for hd in hand_distances.values())
                # Add 10% extra distance to frame hands slightly further out
                max_dist = max_dist * 1.1
                print(f"### [AutoRig] Normalizing hand camera distance to {max_dist:.3f}m (with 10% padding)")
                for hand_suffix in hand_distances:
                    hand_distances[hand_suffix]["dist"] = max_dist

            # Second pass: render with normalized distances
            for hand_suffix in ["L", "R"]:
                print(f"### [AutoRig] Processing hand {hand_suffix}...", flush=True)
                try:
                    if hand_suffix not in hand_distances:
                        print(f"### [AutoRig] Could not frame hand {hand_suffix}, skipping.")
                        continue
                    
                    center = hand_distances[hand_suffix]["center"]
                    dist = hand_distances[hand_suffix]["dist"]
                    size = hand_distances[hand_suffix]["size"]
                    rot_matrix = hand_distances[hand_suffix]["rot_matrix"]

                    # HDRI provides all lighting - no manual lights needed

                    # Include resolution in output for proper keypoint normalization
                    hand_output = {"matrices": [], "image_paths": [], "resolution": res, "center": list(center), "size": size}
                    
                    # 26 views for hand coverage:
                    # - 1 top view
                    # - 16 around horizontal ring at 22.5° increments
                    # - 4 upper diagonal views (45° elevation)
                    # - 4 lower diagonal views (-45° elevation)
                    # - 1 bottom view
                    views = []
                    
                    # Top view
                    views.append(("top", mathutils.Vector((0, 0, dist))))
                    
                    # 16 horizontal ring views at 22.5° increments
                    for i in range(16):
                        angle = math.radians(i * 22.5)
                        offset = mathutils.Vector((math.cos(angle) * dist, math.sin(angle) * dist, 0))
                        views.append((f"side_{i}", offset))
                    
                    # 4 upper diagonal views (45° elevation) at cardinal directions
                    upper_dist = dist * 0.85  # Slightly closer for diagonal views
                    for i, angle in enumerate([0, 90, 180, 270]):
                        angle_rad = math.radians(angle)
                        xy_dist = upper_dist * math.cos(math.radians(45))
                        z_dist = upper_dist * math.sin(math.radians(45))
                        offset = mathutils.Vector((
                            math.cos(angle_rad) * xy_dist,
                            math.sin(angle_rad) * xy_dist,
                            z_dist
                        ))
                        views.append((f"upper_{i}", offset))
                    
                    # 4 lower diagonal views (-45° elevation) at cardinal directions
                    for i, angle in enumerate([45, 135, 225, 315]):  # Offset from upper views
                        angle_rad = math.radians(angle)
                        xy_dist = upper_dist * math.cos(math.radians(45))
                        z_dist = -upper_dist * math.sin(math.radians(45))
                        offset = mathutils.Vector((
                            math.cos(angle_rad) * xy_dist,
                            math.sin(angle_rad) * xy_dist,
                            z_dist
                        ))
                        views.append((f"lower_{i}", offset))
                    
                    # Bottom view
                    views.append(("bottom", mathutils.Vector((0, 0, -dist))))
                    
                    for i, (view_name, offset) in enumerate(views):
                        cam.location = center + rot_matrix @ offset

                        # Point camera at hand center
                        direction = (center - cam.location).to_track_quat('-Z', 'Y')
                        cam.rotation_euler = direction.to_euler()

                        img_path = os.path.join(cfg['output_dir'], f"hand_{hand_suffix}_{i:02d}.png")
                        
                        bpy.context.scene.render.filepath = img_path
                        with suppress_blender_output():
                            bpy.ops.render.render(write_still=True)
                        
                        # Calculate projection matrix: combines camera intrinsics and extrinsics
                        # P maps 3D world points to 2D image coordinates
                        P = cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get()) @ cam.matrix_world.inverted()
                        
                        hand_output["matrices"].append([list(row) for row in P])
                        hand_output["image_paths"].append(img_path)

                    output_data[hand_suffix] = hand_output
                except Exception as e:
                    print(f"### [AutoRig] Error rendering hand {hand_suffix}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"### [AutoRig] Error in render_hand_views: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # Restore original materials before finishing
            for i, mesh in enumerate(meshes):
                if i in original_materials:
                    restore_original_materials(mesh, original_materials[i])
            
            print(f"### [AutoRig] Finally block - writing metadata", flush=True)
            # Always write the metadata file, even if empty
            with open(cfg['meta_path'], 'w') as f:
                json.dump(output_data if 'output_data' in locals() else {}, f)
            print(f"### [AutoRig] Metadata written successfully", flush=True)
            
    elif cfg['mode'] == 'rig':
        basic_cleanup(main_mesh)
        bpy.ops.object.armature_add(location=(0,0,0)); 
        arm = bpy.context.object; arm.name = "AutoRig_Armature"
        bpy.ops.object.mode_set(mode='EDIT'); eb = arm.data.edit_bones; eb.remove(eb[0])
        
        skel = cfg['skeleton']
        
        # Debug: Check foot keypoints
        foot_keypoints = ["Shin.L", "Shin.R", "Ankle.L", "Ankle.R", "BigToe.L", "BigToe.R", "Heel.L", "Heel.R"]
        available_foot_kp = [kp for kp in foot_keypoints if kp in skel]

        
        # OpenPose gives us points, but we need to create bones BETWEEN points
        # OpenPose naming vs Rigify naming:
        # - "Shoulder.L" keypoint -> becomes "upper_arm.L" bone head
        # - "Arm.L" keypoint -> becomes "upper_arm.L" bone tail / "forearm.L" bone head
        # - "Forearm.L" keypoint -> becomes "forearm.L" bone tail / "hand.L" bone head
        
        # Create proper bone structure
        bone_data = {}
        created = {}
        
        # LIST OF KEYPOINTS TO NOT CREATE AS BONES
        FILTER_OUT = ["Nose", "Ear.L", "Ear.R"]
        
        # Synthesize proper anatomical bones from keypoints
        # Root bone - use Hips keypoint as root, extends upward slightly (base for pelvis and spine.001)
        if "Hips" in skel:
            bone = eb.new("root")
            bone.head = skel["Hips"]
            # Root tail should be a short bone, just above hips to anchor pelvis and spine
            if "Spine" in skel:
                # Root extends 1/3 of the way from Hips to Spine (short anchor bone)
                bone.tail = [
                    skel["Hips"][i] + (skel["Spine"][i] - skel["Hips"][i]) * 0.33
                    for i in range(3)
                ]
                print(f"### [AutoRig] Root bone: hips at {bone.head}, extends to {bone.tail}", flush=True)
            else:
                # Fallback: point straight up 5cm
                bone.tail = [skel["Hips"][0], skel["Hips"][1], skel["Hips"][2] + 0.05]
                print(f"### [AutoRig] Root bone (fallback): hips at {bone.head}, extends up 5cm to {bone.tail}", flush=True)
            created["root"] = bone
        
        # Spine.001 (lower spine/lumbar) - from root.tail to Spine keypoint
        if "Spine" in skel and "root" in created:
            bone = eb.new("spine.001")
            bone.head = list(created["root"].tail)  # Start where root ends
            bone.tail = skel["Spine"]  # Extend to Spine keypoint (mid-torso)
            print(f"### [AutoRig] Spine.001: starts at root.tail {bone.head}, extends to Spine keypoint {bone.tail}", flush=True)
            created["spine.001"] = bone
        elif "Spine" in skel and "Hips" in skel:
            bone = eb.new("spine.001")
            bone.head = skel["Hips"]
            bone.tail = skel["Spine"]
            print(f"### [AutoRig] Spine.001 (no root): hips {bone.head} to spine {bone.tail}", flush=True)
            created["spine.001"] = bone
        
        # Spine.002 (mid/upper spine) - from Spine keypoint to midpoint between Spine and Neck
        if "spine.001" in created and "Spine" in skel and "Neck" in skel:
            midpoint = [
                (skel["Spine"][i] + skel["Neck"][i]) / 2
                for i in range(3)
            ]
            bone = eb.new("spine.002")
            bone.head = list(created["spine.001"].tail)  # Start at Spine keypoint
            bone.tail = midpoint
            print(f"### [AutoRig] Spine.002: starts at Spine {bone.head}, extends to midpoint {bone.tail}", flush=True)
            created["spine.002"] = bone
        
        # Spine.003 (upper spine/scapula level) - from midpoint to Neck keypoint
        if "spine.002" in created and "Neck" in skel:
            bone = eb.new("spine.003")
            bone.head = list(created["spine.002"].tail)
            # Spine.003 reaches up to Neck keypoint (base of neck/shoulder level)
            bone.tail = skel["Neck"]
            print(f"### [AutoRig] Spine.003: starts at midpoint {bone.head}, extends to Neck {bone.tail}", flush=True)
            created["spine.003"] = bone
        
        # Neck - from Neck keypoint (base of neck) to chin level
        # Head keypoint is at top of skull, neck should end at chin level
        if "Neck" in skel and "Head" in skel:
            bone = eb.new("neck")
            # Neck starts at Neck keypoint (base of neck/top of shoulders)
            bone.head = skel["Neck"]
            
            # Neck should end at chin level (where head pivots)
            # Priority 1: Use DWPose Face.8 (chin) if available
            if "Face.8" in skel:
                chin_level = skel["Face.8"]
            # Priority 2: Calculate from ears
            elif "Ear.L" in skel and "Ear.R" in skel:
                ear_l = skel["Ear.L"]
                ear_r = skel["Ear.R"]
                # Chin is typically 10-12cm below ear level
                chin_offset = 0.11  # 11cm below ears
                chin_level = [
                    (ear_l[0] + ear_r[0]) / 2,
                    (ear_l[1] + ear_r[1]) / 2,
                    (ear_l[2] + ear_r[2]) / 2 - chin_offset
                ]
            # Priority 3: Estimate from nose
            elif "Nose" in skel:
                nose = skel["Nose"]
                # Chin is typically 8-9cm below nose
                chin_offset = 0.085  # 8.5cm below nose
                chin_level = [nose[0], nose[1], nose[2] - chin_offset]
            # Priority 4: Fallback - 80% up from neck to head top
            else:
                neck_start = bone.head
                head_top = skel["Head"]
                chin_level = [
                    neck_start[0] + (head_top[0] - neck_start[0]) * 0.8,
                    neck_start[1] + (head_top[1] - neck_start[1]) * 0.8,
                    neck_start[2] + (head_top[2] - neck_start[2]) * 0.8
                ]
            
            bone.tail = chin_level
            created["neck"] = bone
        
        # Head - from chin level to highest vertex above the neck
        if "Head" in skel and "neck" in created:
            bone = eb.new("head")
            # Start where neck ends (chin level)
            bone.head = list(created["neck"].tail)
            
            # Find the highest vertex above the neck bone to determine head top
            # For maximum reliability: find absolute highest point above neck (no XY restrictions)
            neck_z = created["neck"].tail[2]
            
            # Find highest vertex anywhere above neck - no radius limit
            highest_z = neck_z
            highest_vertex_pos = None
            vertices_checked = 0
            vertices_above_neck = 0
            
            for v in main_mesh.data.vertices:
                v_world = main_mesh.matrix_world @ v.co
                vertices_checked += 1
                
                # Consider ALL vertices above neck level
                if v_world[2] > neck_z:
                    vertices_above_neck += 1
                    if v_world[2] > highest_z:
                        highest_z = v_world[2]
                        highest_vertex_pos = v_world.copy()
            
            head_height = highest_z - neck_z if highest_vertex_pos else 0
            print(f"### [AutoRig] Head search: checked {vertices_checked} vertices, {vertices_above_neck} above neck, head height={head_height:.4f}m", flush=True)
            
            # If no vertices found above neck, use Head keypoint as fallback
            if highest_vertex_pos is None and "Head" in skel:
                highest_z = skel["Head"][2]
                print(f"### [AutoRig] Head bone: Using Head keypoint as fallback Z={highest_z:.4f}", flush=True)
            
            # IMPORTANT: Keep bone vertically aligned - use neck XY but highest Z
            # This prevents the bone from tilting toward back of hair/bun
            bone.tail = [created["neck"].tail[0], created["neck"].tail[1], highest_z]
            print(f"### [AutoRig] Head bone: chin at {bone.head}, extends VERTICALLY to Z={highest_z:.4f} (height={head_height:.4f}m)", flush=True)
            created["head"] = bone
        elif "Head" in skel:
            # Fallback if no neck: use Head keypoint
            bone = eb.new("head")
            if "Ear.L" in skel and "Ear.R" in skel:
                ear_l = skel["Ear.L"]
                ear_r = skel["Ear.R"]
                base_of_head = [
                    (ear_l[0] + ear_r[0]) / 2,
                    (ear_l[1] + ear_r[1]) / 2,
                    (ear_l[2] + ear_r[2]) / 2
                ]
                bone.head = base_of_head
            else:
                # Estimate: 10cm below head top
                bone.head = [skel["Head"][0], skel["Head"][1], skel["Head"][2] - 0.1]
            bone.tail = skel["Head"]
            created["head"] = bone
        
        # Arms (both sides)
        for side, suffix in [("L", ".L"), ("R", ".R")]:
            # Shoulder/Clavicle - from scapula level (spine.003 tail) to Shoulder keypoint
            if f"Shoulder{suffix}" in skel:
                bone = eb.new(f"shoulder{suffix}")
                # Start from scapula level (where spine.003 ends)
                if "spine.003" in created:
                    bone.head = list(created["spine.003"].tail)
                elif "spine.002" in created:
                    bone.head = list(created["spine.002"].tail)
                elif "spine.001" in created:
                    bone.head = list(created["spine.001"].tail)
                elif "Spine" in skel and "Neck" in skel:
                    # Calculate bottom of neck
                    bone.head = [(skel["Spine"][i] + (skel["Neck"][i] - skel["Spine"][i]) * 0.67) for i in range(3)]
                elif "Neck" in skel:
                    bone.head = skel["Neck"]
                else:
                    continue
                bone.tail = skel[f"Shoulder{suffix}"]
                created[f"shoulder{suffix}"] = bone
            
            # Upper Arm - from Shoulder keypoint to Arm keypoint (elbow)
            if f"Shoulder{suffix}" in skel and f"Arm{suffix}" in skel:
                bone = eb.new(f"upper_arm{suffix}")
                bone.head = skel[f"Shoulder{suffix}"]
                bone.tail = skel[f"Arm{suffix}"]
                created[f"upper_arm{suffix}"] = bone
            
            # Forearm - from Arm keypoint (elbow) to Forearm keypoint (wrist)
            if f"Arm{suffix}" in skel and f"Forearm{suffix}" in skel:
                bone = eb.new(f"forearm{suffix}")
                bone.head = skel[f"Arm{suffix}"]
                bone.tail = skel[f"Forearm{suffix}"]
                created[f"forearm{suffix}"] = bone
            
            # Hand - from Forearm keypoint (wrist from body) to MediaPipe wrist or middle finger base
            if f"Forearm{suffix}" in skel:
                bone = eb.new(f"hand{suffix}")
                bone.head = skel[f"Forearm{suffix}"]
                
                # Use MediaPipe refined wrist position if available (Hand.L or Hand.R from hand refinement)
                mediapipe_wrist = f"Hand{suffix}"
                if mediapipe_wrist in skel:
                    bone.tail = skel[mediapipe_wrist]

                # Fallback to middle finger base
                elif f"Middle.1{suffix}" in skel:
                    bone.tail = skel[f"Middle.1{suffix}"]
                # Last resort: extend 8cm from wrist
                else:
                    bone.tail = [skel[f"Forearm{suffix}"][0], skel[f"Forearm{suffix}"][1], skel[f"Forearm{suffix}"][2] + 0.08]
                created[f"hand{suffix}"] = bone
        
        # Legs (both sides)
        # NOTE: DWPose keypoint names are confusing:
        # - "Thigh" keypoint is AT the hip joint
        # - "Shin" keypoint is AT the knee 
        # - "Ankle" keypoint is AT the ankle
        # We create bones BETWEEN these points
        for side, suffix in [("L", ".L"), ("R", ".R")]:
            # Pelvis bone - synthesized to connect root to hip joint
            # Bridges from center (root) to where thigh starts (hip socket)
            if "Hips" in skel and f"Thigh{suffix}" in skel:
                bone = eb.new(f"pelvis{suffix}")
                bone.head = skel["Hips"]  # At root/center
                bone.tail = skel[f"Thigh{suffix}"]  # At hip joint
                created[f"pelvis{suffix}"] = bone

            
            # Thigh bone - from HIP (Thigh keypoint) to KNEE (Shin keypoint)
            if f"Thigh{suffix}" in skel and f"Shin{suffix}" in skel:
                bone = eb.new(f"thigh{suffix}")
                bone.head = skel[f"Thigh{suffix}"]  # At hip
                bone.tail = skel[f"Shin{suffix}"]    # At knee
                created[f"thigh{suffix}"] = bone

            else:
                print(f"### [AutoRig] Cannot create thigh{suffix}: Thigh{suffix} in skel={f'Thigh{suffix}' in skel}, Shin{suffix} in skel={f'Shin{suffix}' in skel}", flush=True)
            
            # Shin bone - from KNEE (Shin keypoint) down to ground level (lowest foot point)
            if f"Shin{suffix}" in skel and f"Ankle{suffix}" in skel and f"BigToe{suffix}" in skel:
                bone = eb.new(f"shin{suffix}")
                bone.head = skel[f"Shin{suffix}"]    # At knee
                
                # Find the lowest Z point among ankle, heel, and toe (ground level)
                ankle = skel[f"Ankle{suffix}"]
                toe = skel[f"BigToe{suffix}"]
                heel = skel.get(f"Heel{suffix}")
                
                # Get minimum Z (lowest point on ground)
                ground_heights = [ankle[2], toe[2]]
                if heel:
                    ground_heights.append(heel[2])
                ground_level = min(ground_heights)
                
                # Position shin tail at ankle's XY position but at ground level
                bone.tail = [ankle[0], ankle[1], ground_level]
                created[f"shin{suffix}"] = bone
            else:
                print(f"### [AutoRig] Cannot create shin{suffix}: missing keypoints", flush=True)
            
            # Foot bone - from ground level to ball/forward position (horizontal on ground)
            if f"shin{suffix}" in created and f"Heel{suffix}" in skel and f"BigToe{suffix}" in skel:
                bone = eb.new(f"foot{suffix}")
                # Foot starts where shin ends (at ground level)
                bone.head = list(created[f"shin{suffix}"].tail)
                
                heel = skel[f"Heel{suffix}"]
                toe = skel[f"BigToe{suffix}"]
                ground_level = bone.head[2]  # Same Z as shin tail
                
                # Ball position: forward from heel toward toe, at ground level
                ball_pos = [
                    heel[0] + (toe[0] - heel[0]) * 0.7,
                    heel[1] + (toe[1] - heel[1]) * 0.7,
                    ground_level
                ]
                bone.tail = ball_pos
                created[f"foot{suffix}"] = bone
            elif f"shin{suffix}" in created and f"BigToe{suffix}" in skel:
                # Fallback: no Heel keypoint
                bone = eb.new(f"foot{suffix}")
                bone.head = list(created[f"shin{suffix}"].tail)
                toe = skel[f"BigToe{suffix}"]
                ground_level = bone.head[2]
                ball_pos = [
                    bone.head[0] + (toe[0] - bone.head[0]) * 0.7,
                    bone.head[1] + (toe[1] - bone.head[1]) * 0.7,
                    ground_level
                ]
                bone.tail = ball_pos
                created[f"foot{suffix}"] = bone
            else:
                print(f"### [AutoRig] WARNING: Cannot create foot{suffix}, missing keypoints")
            
            # Toe bone - from BALL OF FOOT to TOE TIP
            if f"foot{suffix}" in created and f"BigToe{suffix}" in skel:
                bone = eb.new(f"toe{suffix}")
                bone.head = list(created[f"foot{suffix}"].tail)
                bone.tail = skel[f"BigToe{suffix}"]
                created[f"toe{suffix}"] = bone
        
        # Fingers
        for side, suffix in [("L", ".L"), ("R", ".R")]:
            for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                for i in range(1, 4):
                    bone_name = f"{finger.lower()}.{i:02d}{suffix}"
                    kp_name = f"{finger}.{i}{suffix}"
                    
                    if kp_name in skel:
                        bone = eb.new(bone_name)
                        bone.head = skel[kp_name]
                        
                        # Tail points to next segment or extends in the direction of the finger
                        if i < 3 and f"{finger}.{i+1}{suffix}" in skel:
                            bone.tail = skel[f"{finger}.{i+1}{suffix}"]
                        elif i == 3 and f"{finger}.Tip{suffix}" in skel:
                            # Last bone segment: use the fingertip position for tail
                            bone.tail = skel[f"{finger}.Tip{suffix}"]
                        else:
                            # Fallback: calculate direction from previous segments
                            if i >= 2:
                                # Use direction from previous bone segment
                                prev_kp = f"{finger}.{i-1}{suffix}"
                                if prev_kp in skel:
                                    direction = mathutils.Vector(skel[kp_name]) - mathutils.Vector(skel[prev_kp])
                                    bone.tail = list(mathutils.Vector(skel[kp_name]) + direction.normalized() * 0.02)
                                else:
                                    # Fallback: extend along hand forward direction (Y-axis)
                                    bone.tail = [skel[kp_name][0], skel[kp_name][1] + 0.02, skel[kp_name][2]]
                            else:
                                # First segment with no next - shouldn't happen but handle it
                                bone.tail = [skel[kp_name][0], skel[kp_name][1] + 0.02, skel[kp_name][2]]
                        
                        created[bone_name] = bone
        
        # Set up hierarchy
        hierarchy = [
            ("root", "spine.001"),
            ("spine.001", "spine.002"),
            ("spine.002", "spine.003"),
            ("spine.003", "neck"),
            ("neck", "head"),
            # Shoulders branch from spine.003 (upper spine/scapula), not neck
            ("spine.003", "shoulder.L"), ("shoulder.L", "upper_arm.L"), ("upper_arm.L", "forearm.L"), ("forearm.L", "hand.L"),
            ("spine.003", "shoulder.R"), ("shoulder.R", "upper_arm.R"), ("upper_arm.R", "forearm.R"), ("forearm.R", "hand.R"),
            # Legs: root → pelvis → thigh → shin → foot → toe
            ("root", "pelvis.L"), ("pelvis.L", "thigh.L"), ("thigh.L", "shin.L"), ("shin.L", "foot.L"), ("foot.L", "toe.L"),
            ("root", "pelvis.R"), ("pelvis.R", "thigh.R"), ("thigh.R", "shin.R"), ("shin.R", "foot.R"), ("foot.R", "toe.R"),
        ]
        
        # Define which bones should use connected (continuous chains only, not branches)
        connected_chains = {
            ("root", "spine.001"),
            ("spine.001", "spine.002"),
            ("spine.002", "spine.003"),
            ("spine.003", "neck"),
            ("neck", "head"),
            ("shoulder.L", "upper_arm.L"), ("upper_arm.L", "forearm.L"), ("forearm.L", "hand.L"),
            ("shoulder.R", "upper_arm.R"), ("upper_arm.R", "forearm.R"), ("forearm.R", "hand.R"),
            # Leg chains: pelvis branches from root, then continuous chain down
            ("pelvis.L", "thigh.L"), ("thigh.L", "shin.L"), ("shin.L", "foot.L"), ("foot.L", "toe.L"),
            ("pelvis.R", "thigh.R"), ("thigh.R", "shin.R"), ("shin.R", "foot.R"), ("foot.R", "toe.R"),
        }
        
        # Finger hierarchy
        for side, suffix in [("L", ".L"), ("R", ".R")]:
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                hierarchy.append((f"hand{suffix}", f"{finger}.01{suffix}"))
                hierarchy.append((f"{finger}.01{suffix}", f"{finger}.02{suffix}"))
                hierarchy.append((f"{finger}.02{suffix}", f"{finger}.03{suffix}"))
                # Fingers should NOT be connected to hand bone (they branch from palm)
                # Only finger segments should be connected to each other
                connected_chains.add((f"{finger}.01{suffix}", f"{finger}.02{suffix}"))
                connected_chains.add((f"{finger}.02{suffix}", f"{finger}.03{suffix}"))
        
        # Apply hierarchy
        for parent_name, child_name in hierarchy:
            if parent_name in created and child_name in created:
                created[child_name].parent = created[parent_name]
                # Only use connected bones for continuous chains, not branches
                if (parent_name, child_name) in connected_chains:
                    created[child_name].use_connect = True
                    
        # Debug: Print root and thigh bone positions


        # Fix end bones (bones with no children) to have proper tails
        # EXCLUDE head bone - it already has correct tail position set to top of mesh
        for b in eb:
            if not b.children and b.parent and b.name.lower() != "head":
                # Extend tail in same direction as parent bone
                vec = (mathutils.Vector(b.head) - mathutils.Vector(b.parent.head)).normalized()
                b.tail = mathutils.Vector(b.head) + (vec * 0.02)
        
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        fix_bone_orientations(arm)
        
        # DISABLED: Mesh fitting overrides triangulation and produces worse results
        # # Fit finger bones to mesh geometry (find actual finger cylinders)
        # print("### [AutoRig] Attempting to fit finger bones to mesh geometry...", flush=True)
        # try:
        #     fit_fingers_to_mesh(arm, main_mesh)
        # except Exception as e:
        #     print(f"### [AutoRig] Finger fitting failed: {e}", flush=True)
        #     import traceback
        #     traceback.print_exc()
        
        bpy.ops.object.select_all(action='DESELECT')
        for mesh in meshes:
            mesh.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        
        # Check weighting for all meshes
        all_failed_bones = []
        for mesh in meshes:
            failed_bones = []
            for b in arm.data.bones:
                vg = mesh.vertex_groups.get(b.name)
                if not vg or len([v for v in mesh.data.vertices if any(g.group == vg.index for g in v.groups)]) == 0:
                    failed_bones.append(b.name)
            
            if failed_bones:
                print(f"### [AutoRig] Initial Bone Heat failed for {len(failed_bones)} bones on mesh '{mesh.name}'")
                all_failed_bones.extend(failed_bones)
        
        if all_failed_bones:
            # Try recovery on problematic meshes
            for mesh in meshes:
                recover_broken_mesh(mesh)
            
            bpy.ops.object.select_all(action='DESELECT')
            for mesh in meshes:
                mesh.select_set(True)
            arm.select_set(True)
            bpy.context.view_layer.objects.active = arm
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
            
            still_failed = []
            for mesh in meshes:
                for b in arm.data.bones:
                    vg = mesh.vertex_groups.get(b.name)
                    if not vg or len([v for v in mesh.data.vertices if any(g.group == vg.index for g in v.groups)]) == 0:
                        still_failed.append(b.name)
            
            if still_failed:
                print(f"### [AutoRig] Recovery failed. Final fallback to Envelope.")
                bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
                for mesh in meshes:
                    bpy.context.view_layer.objects.active = mesh
                    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
                    bpy.ops.object.vertex_group_smooth(factor=0.6, repeat=8)
                    bpy.ops.object.mode_set(mode='OBJECT')

        # HumanIK bone renaming if requested
        humanik_ready = cfg.get('humanik_ready', False)
        if humanik_ready:
            print(f"### [AutoRig] Applying HumanIK bone naming...")
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # HumanIK standard bone names mapping
            humanik_map = {
                "root": "Hips",
                "pelvis.L": "LeftUpLeg",
                "pelvis.R": "RightUpLeg",
                "thigh.L": "LeftLeg",
                "thigh.R": "RightLeg",
                "shin.L": "LeftFoot",  # Maya HumanIK uses Foot for shin
                "shin.R": "RightFoot",
                "foot.L": "LeftToeBase",
                "foot.R": "RightToeBase",
                "toe.L": "LeftToe",
                "toe.R": "RightToe",
                "spine.001": "Spine",
                "spine.002": "Spine1",
                "spine.003": "Spine2",
                "neck": "Neck",
                "head": "Head",
                "shoulder.L": "LeftShoulder",
                "shoulder.R": "RightShoulder",
                "upper_arm.L": "LeftArm",
                "upper_arm.R": "RightArm",
                "forearm.L": "LeftForeArm",
                "forearm.R": "RightForeArm",
                "hand.L": "LeftHand",
                "hand.R": "RightHand",
                # Fingers - Maya HumanIK standard naming
                "thumb.01.L": "LeftHandThumb1", "thumb.02.L": "LeftHandThumb2", "thumb.03.L": "LeftHandThumb3",
                "thumb.01.R": "RightHandThumb1", "thumb.02.R": "RightHandThumb2", "thumb.03.R": "RightHandThumb3",
                "index.01.L": "LeftHandIndex1", "index.02.L": "LeftHandIndex2", "index.03.L": "LeftHandIndex3",
                "index.01.R": "RightHandIndex1", "index.02.R": "RightHandIndex2", "index.03.R": "RightHandIndex3",
                "middle.01.L": "LeftHandMiddle1", "middle.02.L": "LeftHandMiddle2", "middle.03.L": "LeftHandMiddle3",
                "middle.01.R": "RightHandMiddle1", "middle.02.R": "RightHandMiddle2", "middle.03.R": "RightHandMiddle3",
                "ring.01.L": "LeftHandRing1", "ring.02.L": "LeftHandRing2", "ring.03.L": "LeftHandRing3",
                "ring.01.R": "RightHandRing1", "ring.02.R": "RightHandRing2", "ring.03.R": "RightHandRing3",
                "pinky.01.L": "LeftHandPinky1", "pinky.02.L": "LeftHandPinky2", "pinky.03.L": "LeftHandPinky3",
                "pinky.01.R": "RightHandPinky1", "pinky.02.R": "RightHandPinky2", "pinky.03.R": "RightHandPinky3",
            }
            
            # Rename bones for HumanIK
            for old_name, new_name in humanik_map.items():
                if old_name in arm.data.bones:
                    arm.data.bones[old_name].name = new_name
            
            print(f"### [AutoRig] HumanIK bone naming applied")
        
        # Set armature origin to match original model pivot
        # Calculate the lowest Z point (ground/feet level) across all meshes
        min_z = float('inf')
        for mesh in meshes:
            for v in mesh.bound_box:
                world_v = mesh.matrix_world @ mathutils.Vector(v)
                if world_v[2] < min_z:
                    min_z = world_v[2]
        
        # Set armature origin to ground level at XY center
        print(f"### [AutoRig] Setting armature origin to ground level (Z={min_z:.3f})", flush=True)
        bpy.context.scene.cursor.location = (0, 0, min_z)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        
        bpy.ops.object.select_all(action='DESELECT')
        for mesh in meshes:
            mesh.select_set(True)
        arm.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Export settings optimized for Maya
        export_axis = '-Z' if not humanik_ready else 'Y'  # Maya uses Y-up
        
        bpy.ops.export_scene.fbx(
            filepath=cfg['export_path'], 
            use_selection=True, 
            object_types={'ARMATURE', 'MESH'}, 
            mesh_smooth_type='FACE', 
            axis_forward=export_axis, 
            axis_up='Y',
            add_leaf_bones=True,
            bake_anim=False,
            primary_bone_axis='Y',
            secondary_bone_axis='X',
            apply_unit_scale=True,
            apply_scale_options='FBX_SCALE_ALL'
        )
        
        if humanik_ready:
            print(f"### [AutoRig] Exported HumanIK-ready FBX - ready for Maya Character Definition")

    elif cfg.get('mode') == 'adapt_universal_rig':
        print(f"### [UniversalRig] Adapting AI2Mesh universal rig to detected skeleton...", flush=True)

        # Deterministic mapping: AI2Mesh bone name → skeleton joint key
        # Each entry: bone_tail will be placed at skel[value], bone_head from parent's mapped key
        AI2MESH_TO_SKEL = {
            # Root/Hip chain — tail points toward the spine so the bone
            # direction is upward from pelvis.  Head is mapped in HEAD_MAP.
            "CC_Base_Hip":    "Spine",

            # Spine chain (only map key anchors; intermediate bones stay at template proportions)
            "CC_Base_Neck":   "Neck",
            "CC_Base_Head":   "Head",

            # Left arm chain (linear chain — all joints map cleanly)
            "CC_Base_L_Clavicle": "Shoulder.L",
            "CC_Base_L_Upperarm": "Arm.L",
            "CC_Base_L_Forearm":  "Forearm.L",
            "CC_Base_L_Hand":     "Hand.L",

            # Right arm chain
            "CC_Base_R_Clavicle": "Shoulder.R",
            "CC_Base_R_Upperarm": "Arm.R",
            "CC_Base_R_Forearm":  "Forearm.R",
            "CC_Base_R_Hand":     "Hand.R",

            # Left leg  (DWPose: "Thigh"=hip socket, "Shin"=knee, "Ankle"=ankle)
            "CC_Base_L_Thigh": "Shin.L",    # thigh tail → knee
            "CC_Base_L_Calf":  "Ankle.L",   # calf tail → ankle
            "CC_Base_L_Foot":  "BigToe.L",  # foot tail → ball of foot

            # Right leg
            "CC_Base_R_Thigh": "Shin.R",
            "CC_Base_R_Calf":  "Ankle.R",
            "CC_Base_R_Foot":  "BigToe.R",

            # Left fingers — each CC bone is a phalanx; tail maps to the
            # DISTAL end (the next joint down the chain).  Head positions for
            # the first bone of each finger are set in AI2MESH_HEAD_MAP.
            "CC_Base_L_Thumb1": "Thumb.2.L",
            "CC_Base_L_Thumb2": "Thumb.3.L",
            "CC_Base_L_Thumb3": "Thumb.Tip.L",
            "CC_Base_L_Index1": "Index.2.L",
            "CC_Base_L_Index2": "Index.3.L",
            "CC_Base_L_Index3": "Index.Tip.L",
            "CC_Base_L_Mid1":   "Middle.2.L",
            "CC_Base_L_Mid2":   "Middle.3.L",
            "CC_Base_L_Mid3":   "Middle.Tip.L",
            "CC_Base_L_Ring1":  "Ring.2.L",
            "CC_Base_L_Ring2":  "Ring.3.L",
            "CC_Base_L_Ring3":  "Ring.Tip.L",
            "CC_Base_L_Pinky1": "Pinky.2.L",
            "CC_Base_L_Pinky2": "Pinky.3.L",
            "CC_Base_L_Pinky3": "Pinky.Tip.L",

            # Right fingers
            "CC_Base_R_Thumb1": "Thumb.2.R",
            "CC_Base_R_Thumb2": "Thumb.3.R",
            "CC_Base_R_Thumb3": "Thumb.Tip.R",
            "CC_Base_R_Index1": "Index.2.R",
            "CC_Base_R_Index2": "Index.3.R",
            "CC_Base_R_Index3": "Index.Tip.R",
            "CC_Base_R_Mid1":   "Middle.2.R",
            "CC_Base_R_Mid2":   "Middle.3.R",
            "CC_Base_R_Mid3":   "Middle.Tip.R",
            "CC_Base_R_Ring1":  "Ring.2.R",
            "CC_Base_R_Ring2":  "Ring.3.R",
            "CC_Base_R_Ring3":  "Ring.Tip.R",
            "CC_Base_R_Pinky1": "Pinky.2.R",
            "CC_Base_R_Pinky2": "Pinky.3.R",
            "CC_Base_R_Pinky3": "Pinky.Tip.R",
        }

        try:
            skel = cfg.get('skeleton') or cfg.get('skeleton_3d')
            if not skel or not isinstance(skel, dict):
                print(f"### [UniversalRig] No skeleton provided — exporting template as-is", flush=True)
            else:
                print(f"### [UniversalRig] Skeleton has {len(skel)} joints: {list(skel.keys())[:10]}...", flush=True)

            # ----------------------------------------------------------------
            # Stage A — Import target mesh to measure actual scale
            # The skeleton is in native post-transform_apply space (no
            # normalisation).  We need to convert from skeleton space to the
            # target mesh's actual import dimensions.
            # ----------------------------------------------------------------
            target_actual_height = 1.8  # default: assume already 1.8 m
            target_center_x = 0.0
            target_center_y = 0.0
            target_min_z = 0.0
            # Track ALL objects imported with the target (meshes + armatures + empties)
            # so we can cleanly remove them all before export.
            target_imported_names = set()

            model_path = cfg.get('model_path', '')
            if skel and isinstance(skel, dict) and model_path and os.path.exists(model_path):
                print(f"### [UniversalRig] Importing target mesh for bounds measurement: {model_path}", flush=True)
                before_target = set(o.name for o in bpy.data.objects)
                ext = os.path.splitext(model_path)[1].lower()
                try:
                    if ext == '.fbx':
                        bpy.ops.import_scene.fbx(filepath=model_path, use_anim=False)
                    elif ext == '.obj':
                        bpy.ops.wm.obj_import(filepath=model_path)
                    elif ext in ('.glb', '.gltf'):
                        bpy.ops.import_scene.gltf(filepath=model_path)
                    else:
                        print(f"### [UniversalRig] Unsupported format '{ext}', using defaults", flush=True)
                except Exception as import_err:
                    print(f"### [UniversalRig] Target import failed: {import_err} — using defaults", flush=True)

                # Track ALL new objects (meshes, armatures, empties — everything)
                target_imported_names = set(o.name for o in bpy.data.objects) - before_target

                all_verts = []
                for n in target_imported_names:
                    obj = bpy.data.objects.get(n)
                    if obj and obj.type == 'MESH':
                        for v in obj.data.vertices:
                            all_verts.append(obj.matrix_world @ v.co)
                if all_verts:
                    xs = [v.x for v in all_verts]
                    ys = [v.y for v in all_verts]
                    zs = [v.z for v in all_verts]
                    target_min_z = min(zs)
                    target_center_x = (max(xs) + min(xs)) / 2.0
                    target_center_y = (max(ys) + min(ys)) / 2.0
                    measured_height = max(zs) - min(zs)
                    if measured_height > 0.01:
                        target_actual_height = measured_height
                    print(
                        f"### [UniversalRig] Target: height={target_actual_height:.3f} BU "
                        f"center=({target_center_x:.3f},{target_center_y:.3f}) "
                        f"min_z={target_min_z:.3f}",
                        flush=True
                    )
            else:
                print(f"### [UniversalRig] No model_path — using defaults (1.8 m)", flush=True)

            # Skeleton ground Z: feet level in normalised skel space
            skel_ground_z = 0.0
            if skel:
                fz = [skel[k][2] for k in ("Heel.L", "Heel.R", "Ankle.L", "Ankle.R") if k in skel]
                if fz:
                    skel_ground_z = min(fz)

            skel_zs = [pos[2] for pos in skel.values() if hasattr(pos, '__len__') and len(pos) >= 3]
            skel_height = (max(skel_zs) - min(skel_zs)) if skel_zs else 0.0
            skel_to_actual = target_actual_height / skel_height if skel_height > 0.01 else 1.0
            print(f"### [UniversalRig] skel_to_actual={skel_to_actual:.4f} (skel_height={skel_height:.3f}, target={target_actual_height:.3f})", flush=True)

            def skel_to_world(pos):
                """Skeleton space position → actual world position."""
                return mathutils.Vector((
                    target_center_x + pos[0] * skel_to_actual,
                    target_center_y + pos[1] * skel_to_actual,
                    target_min_z + (pos[2] - skel_ground_z) * skel_to_actual,
                ))

            # ----------------------------------------------------------------
            # Import the AI2Mesh template rig, track new objects
            # ----------------------------------------------------------------
            template = cfg.get('universal_template')
            if not template or not os.path.exists(template):
                print(f"### [UniversalRig] ERROR: Template not found at {template}", flush=True)
                return
            print(f"### [UniversalRig] Importing template: {template}", flush=True)
            before_template = set(o.name for o in bpy.data.objects)
            bpy.ops.import_scene.fbx(filepath=template, use_anim=False)
            template_obj_names = set(o.name for o in bpy.data.objects) - before_template

            modifications = []

            if skel and isinstance(skel, dict):
                # ----------------------------------------------------------------
                # Stage B — Scale the template so its mesh bounds match the target
                # mesh bounds, then position it at the target location.
                # Applying the transform BEFORE bone work ensures POSE MODE
                # matrices are in the correct world space.
                # ----------------------------------------------------------------
                arm = next(
                    (o for o in bpy.data.objects
                     if o.type == 'ARMATURE' and o.name in template_obj_names),
                    None
                )
                if arm is None:
                    print("### [UniversalRig] ERROR: No armature found in template", flush=True)
                    return

                # Measure template mesh bounds (matrix_world already includes
                # any import-level scale, so we get correct world-space values
                # without needing to bake / apply the object scale first).
                tmpl_verts = []
                for n in template_obj_names:
                    obj = bpy.data.objects.get(n)
                    if obj and obj.type == 'MESH':
                        for v in obj.data.vertices:
                            tmpl_verts.append(obj.matrix_world @ v.co)

                if tmpl_verts:
                    t_zs = [v.z for v in tmpl_verts]
                    t_xs = [v.x for v in tmpl_verts]
                    t_ys = [v.y for v in tmpl_verts]
                    tmpl_height = max(t_zs) - min(t_zs)
                    sf = target_actual_height / tmpl_height if tmpl_height > 0.01 else skel_to_actual
                    print(
                        f"### [UniversalRig] Template height={tmpl_height:.3f}m → scale {sf:.4f}×",
                        flush=True
                    )
                else:
                    sf = skel_to_actual
                    print(f"### [UniversalRig] No template mesh, using skel_to_actual={sf:.4f}", flush=True)

                # Multiply sf into the armature's existing (import) scale.
                # We do NOT apply the transform — the import scale (e.g. 0.01
                # for a cm FBX) must stay on the object so that child meshes
                # and the armature modifier remain in the same coordinate space.
                # arm_mat_inv (computed later) accounts for the non-unit scale
                # when converting world-space skeleton positions to bone-local,
                # and FBX_SCALE_ALL bakes everything on export.
                arm.scale = (arm.scale.x * sf, arm.scale.y * sf, arm.scale.z * sf)
                bpy.context.view_layer.update()
                print(
                    f"### [UniversalRig] Armature scale after sf: "
                    f"({arm.scale.x:.6f}, {arm.scale.y:.6f}, {arm.scale.z:.6f})",
                    flush=True
                )

                # Recompute bounds after scaling, then translate to align
                tmpl_verts2 = []
                for n in template_obj_names:
                    obj = bpy.data.objects.get(n)
                    if obj and obj.type == 'MESH':
                        for v in obj.data.vertices:
                            tmpl_verts2.append(obj.matrix_world @ v.co)

                if tmpl_verts2:
                    scaled_min_z = min(v.z for v in tmpl_verts2)
                    scaled_cx = (max(v.x for v in tmpl_verts2) + min(v.x for v in tmpl_verts2)) / 2.0
                    scaled_cy = (max(v.y for v in tmpl_verts2) + min(v.y for v in tmpl_verts2)) / 2.0
                    arm.location.x += target_center_x - scaled_cx
                    arm.location.y += target_center_y - scaled_cy
                    arm.location.z += target_min_z - scaled_min_z
                    bpy.context.view_layer.update()
                    print(
                        f"### [UniversalRig] Template aligned: bottom={target_min_z:.3f} "
                        f"cx={target_center_x:.3f} cy={target_center_y:.3f}",
                        flush=True
                    )

                # ----------------------------------------------------------------
                # Stage C — POSE MODE: set bone directions from skeleton data.
                # Relative skel vectors (scale-invariant) give bone orientations.
                # Head positions use skel_to_world for explicitly mapped bones.
                # armature_apply() bakes the pose → the skinned mesh deforms.
                # ----------------------------------------------------------------
                AI2MESH_HEAD_MAP = {
                    # Hip/pelvis: head at hip center, tail already → Spine
                    "CC_Base_Hip":     "Hips",
                    # Legs
                    "CC_Base_L_Thigh": "Thigh.L",
                    "CC_Base_R_Thigh": "Thigh.R",
                    "CC_Base_L_Calf":  "Shin.L",
                    "CC_Base_R_Calf":  "Shin.R",
                    "CC_Base_L_Foot":  "Ankle.L",
                    "CC_Base_R_Foot":  "Ankle.R",
                    # Head/neck
                    "CC_Base_Head":    "Neck",
                    # Clavicles: head at Neck (sternum level) so the bone
                    # points nearly horizontally to the shoulder.
                    "CC_Base_L_Clavicle": "Neck",
                    "CC_Base_R_Clavicle": "Neck",
                    # First finger bones: head at the proximal (MCP/CMC) joint
                    # so the bone direction is along the phalanx, not from
                    # the wrist (which caused the triangulation artefact).
                    "CC_Base_L_Thumb1":  "Thumb.1.L",
                    "CC_Base_L_Index1":  "Index.1.L",
                    "CC_Base_L_Mid1":    "Middle.1.L",
                    "CC_Base_L_Ring1":   "Ring.1.L",
                    "CC_Base_L_Pinky1":  "Pinky.1.L",
                    "CC_Base_R_Thumb1":  "Thumb.1.R",
                    "CC_Base_R_Index1":  "Index.1.R",
                    "CC_Base_R_Mid1":    "Middle.1.R",
                    "CC_Base_R_Ring1":   "Ring.1.R",
                    "CC_Base_R_Pinky1":  "Pinky.1.R",
                }

                print(f"### [UniversalRig] Repositioning bones in armature: {arm.name}", flush=True)
                bpy.ops.object.select_all(action='DESELECT')
                arm.select_set(True)
                bpy.context.view_layer.objects.active = arm
                bpy.ops.object.mode_set(mode='POSE')

                arm_mat_inv = arm.matrix_world.inverted()
                arm_rot_inv = arm.matrix_world.inverted_safe().to_3x3()

                def bone_depth(pb):
                    d, p = 0, pb.parent
                    while p:
                        d += 1
                        p = p.parent
                    return d

                sorted_pbones = sorted(arm.pose.bones, key=bone_depth)

                def nearest_mapped_ancestor_key(pb):
                    p = pb.parent
                    while p:
                        k = AI2MESH_TO_SKEL.get(p.name)
                        if k:
                            return k
                        p = p.parent
                    return None

                for pbone in sorted_pbones:
                    bone_name = pbone.name
                    tail_key = AI2MESH_TO_SKEL.get(bone_name)
                    if not (tail_key and tail_key in skel):
                        continue

                    # Determine skel FROM/TO points for direction
                    head_key = AI2MESH_HEAD_MAP.get(bone_name)
                    if head_key and head_key in skel:
                        from_skel = mathutils.Vector(skel[head_key])
                    else:
                        anc_key = nearest_mapped_ancestor_key(pbone)
                        if not (anc_key and anc_key in skel):
                            print(f"### [UniversalRig] Skipping {bone_name}: no head ref", flush=True)
                            continue
                        head_key = anc_key
                        from_skel = mathutils.Vector(skel[head_key])

                    to_skel = mathutils.Vector(skel[tail_key])
                    dir_skel = to_skel - from_skel
                    if dir_skel.length < 1e-5:
                        print(f"### [UniversalRig] Skipping {bone_name}: zero direction", flush=True)
                        continue

                    # Bone Y axis: direction in armature space (unit vector)
                    y_axis = (arm_rot_inv @ dir_skel).normalized()

                    # Head position: use skel_to_world for explicitly mapped bones;
                    # for connected bones fall back to current scaled template head
                    # so the connectivity constraints are naturally respected.
                    bpy.context.view_layer.update()
                    if AI2MESH_HEAD_MAP.get(bone_name) in skel:
                        h = arm_mat_inv @ skel_to_world(skel[AI2MESH_HEAD_MAP[bone_name]])
                    else:
                        h = pbone.head.copy()

                    # Y-scale: stretch bone along Y so it reaches the target tail.
                    # armature_apply() bakes non-uniform bone scale by changing
                    # the bone's rest length and deforming the skinned mesh.
                    tail_arm = arm_mat_inv @ skel_to_world(skel[tail_key])
                    desired_len = (tail_arm - h).length
                    rest_len = pbone.bone.length
                    scale_y = desired_len / rest_len if rest_len > 1e-6 else 1.0
                    y_scaled = y_axis * scale_y

                    # Preserve bone roll: project rest X axis perpendicular to new Y
                    rest_mat = pbone.bone.matrix_local
                    old_x = mathutils.Vector((rest_mat[0][0], rest_mat[1][0], rest_mat[2][0]))
                    x_perp = old_x - y_axis * old_x.dot(y_axis)
                    if x_perp.length > 1e-5:
                        x_axis = x_perp.normalized()
                    else:
                        arb = mathutils.Vector((1, 0, 0)) if abs(y_axis.x) < 0.9 else mathutils.Vector((0, 1, 0))
                        x_axis = y_axis.cross(arb).normalized()

                    z_axis = x_axis.cross(y_axis).normalized()
                    x_axis = y_axis.cross(z_axis).normalized()

                    mat = mathutils.Matrix((
                        (x_axis.x, y_scaled.x, z_axis.x, h.x),
                        (x_axis.y, y_scaled.y, z_axis.y, h.y),
                        (x_axis.z, y_scaled.z, z_axis.z, h.z),
                        (0.0,      0.0,        0.0,       1.0),
                    ))
                    try:
                        pbone.matrix = mat
                        bpy.context.view_layer.update()
                        modifications.append({'bone': bone_name, 'tail_key': tail_key, 'head_key': head_key})
                        print(f"### [UniversalRig] Posed {bone_name}: {head_key}→{tail_key}", flush=True)
                    except Exception as e:
                        print(f"### [UniversalRig] Failed to pose {bone_name}: {e}", flush=True)

                # Bake pose as new rest pose so the skinned mesh exports deformed
                print(f"### [UniversalRig] Baking {len(modifications)} bone poses to rest...", flush=True)
                applied = False
                try:
                    with bpy.context.temp_override(
                        object=arm, active_object=arm,
                        selected_objects=[arm],
                    ):
                        bpy.ops.pose.armature_apply(selected=False)
                    applied = True
                    print(f"### [UniversalRig] armature_apply OK (temp_override)", flush=True)
                except Exception as e:
                    print(f"### [UniversalRig] armature_apply temp_override failed: {e}", flush=True)
                if not applied:
                    try:
                        bpy.ops.pose.armature_apply(selected=False)
                        applied = True
                        print(f"### [UniversalRig] armature_apply OK (direct)", flush=True)
                    except Exception as e2:
                        print(f"### [UniversalRig] armature_apply direct also failed: {e2}", flush=True)
                bpy.ops.object.mode_set(mode='OBJECT')
                print(f"### [UniversalRig] Pose baked={applied}, {len(modifications)} bones moved.", flush=True)

                # ----------------------------------------------------------------
                # Stage D — EDIT MODE: set exact bone head/tail positions from
                # the skeleton data.  The pose-mode pass (Stage C) set bone
                # orientations and approximate heads but preserved template bone
                # LENGTHS, so tails didn't reach the target joint positions.
                # armature_apply already baked the deformation into the mesh, so
                # these edit-mode adjustments only affect bone visualisation and
                # the exported rest-pose skeleton — not the mesh shape.
                # ----------------------------------------------------------------
                bpy.ops.object.select_all(action='DESELECT')
                arm.select_set(True)
                bpy.context.view_layer.objects.active = arm
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.context.view_layer.update()
                arm_mat_inv_edit = arm.matrix_world.inverted()

                edit_count = 0
                # Process bones root→leaf so parent tails are set before children
                def ebone_depth(eb):
                    d, p = 0, eb.parent
                    while p:
                        d += 1
                        p = p.parent
                    return d
                sorted_ebones = sorted(arm.data.edit_bones, key=ebone_depth)

                for ebone in sorted_ebones:
                    tail_key = AI2MESH_TO_SKEL.get(ebone.name)
                    if not (tail_key and tail_key in skel):
                        continue

                    # Disconnect so we can position head independently of parent
                    ebone.use_connect = False

                    # --- TAIL: always set from the skeleton joint ---
                    tail_world = skel_to_world(skel[tail_key])
                    ebone.tail = arm_mat_inv_edit @ tail_world

                    # --- HEAD ---
                    head_key = AI2MESH_HEAD_MAP.get(ebone.name)
                    if head_key and head_key in skel:
                        head_world = skel_to_world(skel[head_key])
                        ebone.head = arm_mat_inv_edit @ head_world
                    elif ebone.parent and AI2MESH_TO_SKEL.get(ebone.parent.name):
                        # Chain continuity: snap head to parent's tail
                        ebone.head = ebone.parent.tail.copy()

                    edit_count += 1

                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.context.view_layer.update()
                print(f"### [UniversalRig] Edit-mode: {edit_count} bones positioned head-to-tail.", flush=True)

            # Ensure export directory exists
            export_dir = os.path.dirname(cfg['export_path'])
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)

            # ----------------------------------------------------------------
            # Export 1: Universal template mesh + adapted armature
            # ----------------------------------------------------------------
            # Deselect everything, then select only template objects
            bpy.ops.object.select_all(action='DESELECT')
            for n in template_obj_names:
                obj = bpy.data.objects.get(n)
                if obj:
                    obj.select_set(True)

            bpy.ops.export_scene.fbx(
                filepath=cfg['export_path'],
                use_selection=True,
                object_types={'ARMATURE', 'MESH'},
                axis_forward='-Z',
                axis_up='Y',
                add_leaf_bones=False,
                bake_anim=False,
                apply_unit_scale=True,
                apply_scale_options='FBX_SCALE_ALL',
            )
            print(f"### [UniversalRig] Exported universal rig to {cfg['export_path']}", flush=True)

            # ----------------------------------------------------------------
            # Export 2: Target mesh + duplicated armature (auto-weighted)
            # ----------------------------------------------------------------
            target_export_path = os.path.splitext(cfg['export_path'])[0] + '_target.fbx'

            # Identify target mesh objects and any armatures that came with the target
            target_meshes = []
            target_armatures = []
            for n in target_imported_names:
                obj = bpy.data.objects.get(n)
                if not obj:
                    continue
                if obj.type == 'MESH':
                    target_meshes.append(obj)
                elif obj.type == 'ARMATURE':
                    target_armatures.append(obj)

            if target_meshes:
                # Delete any armatures that came with the target import
                for tarm in target_armatures:
                    bpy.data.objects.remove(tarm, do_unlink=True)
                bpy.context.view_layer.update()

                # Duplicate the adapted armature for the target mesh
                bpy.ops.object.select_all(action='DESELECT')
                arm.select_set(True)
                bpy.context.view_layer.objects.active = arm
                bpy.ops.object.duplicate()
                arm_copy = bpy.context.active_object
                print(f"### [UniversalRig] Duplicated armature: {arm_copy.name}", flush=True)

                # Clear any existing parents/armature modifiers on target meshes
                for tmesh in target_meshes:
                    tmesh.parent = None
                    for mod in list(tmesh.modifiers):
                        if mod.type == 'ARMATURE':
                            tmesh.modifiers.remove(mod)

                # Parent target meshes to duplicated armature with auto-weights
                bpy.ops.object.select_all(action='DESELECT')
                for tmesh in target_meshes:
                    tmesh.select_set(True)
                arm_copy.select_set(True)
                bpy.context.view_layer.objects.active = arm_copy
                try:
                    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
                    print(f"### [UniversalRig] Auto-weighted {len(target_meshes)} target mesh(es) to armature copy", flush=True)
                except Exception as aw_err:
                    print(f"### [UniversalRig] Auto-weight failed: {aw_err} — using envelope weights", flush=True)
                    try:
                        bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
                    except Exception as env_err:
                        print(f"### [UniversalRig] Envelope weight also failed: {env_err}", flush=True)

                # Export target mesh + duplicated armature
                bpy.ops.object.select_all(action='DESELECT')
                for tmesh in target_meshes:
                    tmesh.select_set(True)
                arm_copy.select_set(True)
                bpy.context.view_layer.objects.active = arm_copy

                bpy.ops.export_scene.fbx(
                    filepath=target_export_path,
                    use_selection=True,
                    object_types={'ARMATURE', 'MESH'},
                    axis_forward='-Z',
                    axis_up='Y',
                    add_leaf_bones=False,
                    bake_anim=False,
                    apply_unit_scale=True,
                    apply_scale_options='FBX_SCALE_ALL',
                )
                print(f"### [UniversalRig] Exported target rig to {target_export_path}", flush=True)
            else:
                print(f"### [UniversalRig] No target meshes found — skipping target export", flush=True)

            meta_path = cfg.get('meta_path') or os.path.join(export_dir, 'adapt_meta.json')
            meta = {"modifications": modifications, "modification_count": len(modifications)}
            with open(meta_path, 'w') as mf:
                json.dump(meta, mf)

        except Exception as e:
            import traceback
            print(f"### [UniversalRig] adapt_universal_rig failed: {e}", flush=True)
            traceback.print_exc()

if __name__ == "__main__":
    run_task()