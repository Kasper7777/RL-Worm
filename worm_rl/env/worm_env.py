import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import os
import time

class WormEnv(gym.Env):
    def __init__(self, render=False):
        super(WormEnv, self).__init__()
        
        self.render_mode = render
        self.connected = False
        self.physics_client = None
        
        # Initialize variables
        self.previous_x = 0
        self.previous_height = 0
        self.steps_since_reset = 0
        self.plane_id = None
        self.worm_id = None
        self.food_pos = [2.0, 0.0, 0.025]  # Initialize food position
        self.food_visual_id = None
        self.stairs = []
        self.max_steps = 2000
        
        # Connect to physics engine
        self._connect_physics()
        
        # Get the path to the URDF file
        self.urdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "urdf", "worm.urdf")
        
        # Updated for 5 joints
        self.num_joints = 5
        
        # Define action space (joint velocities)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_joints,), 
            dtype=np.float32
        )
        
        # Define observation space
        obs_dim = (2 * self.num_joints  # joint positions and velocities
                  + 7  # base position (3) and orientation (4)
                  + 3  # food position
                  + 3)  # closest stair position
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Stair parameters
        self.num_stairs = 5
        self.stair_height = 0.05
        self.stair_width = 0.2
        self.stair_depth = 0.2
        
        # Physics parameters for climbing
        self.max_joint_velocity = 1.0
        self.sim_steps_per_action = 4
        self.joint_damping = 0.3
        self.position_gain = 0.5
        self.velocity_gain = 0.5
        self.max_force = 2.0
        self.render_delay = 0.01

    def _connect_physics(self):
        """Safely connect to PyBullet physics server"""
        try:
            # Always try to disconnect first
            if self.connected:
                try:
                    p.disconnect(self.physics_client)
                except:
                    pass
                self.connected = False
                time.sleep(0.1)
            
            # Try to connect multiple times
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    if self.render_mode:
                        self.physics_client = p.connect(p.GUI)
                    else:
                        self.physics_client = p.connect(p.DIRECT)
                    
                    self.connected = True
                    break
                except p.error as e:
                    print(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)
                        continue
                    else:
                        raise
            
            if not self.connected:
                raise Exception("Failed to connect to PyBullet after multiple attempts")
            
            # Configure GUI and physics
            if self.render_mode:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Enable shadows
                
                # Adjust camera for better view
                p.resetDebugVisualizerCamera(
                    cameraDistance=0.7,  # Closer view
                    cameraYaw=0,
                    cameraPitch=-20,
                    cameraTargetPosition=[0, 0, 0]
                )
                time.sleep(0.2)
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1./120.)  # Slower physics timestep
            p.setRealTimeSimulation(0)
            
        except Exception as e:
            print(f"Error connecting to PyBullet: {e}")
            self.connected = False
            raise

    def _spawn_food(self):
        """Spawn food at a random position"""
        try:
            # First try to remove existing food if it exists
            if self.food_visual_id is not None:
                try:
                    p.removeBody(self.food_visual_id)
                except:
                    # If removal fails, just ignore and create new food
                    pass
                finally:
                    self.food_visual_id = None
            
            if not self.connected or self.worm_id is None:
                print("Warning: Cannot spawn food - not connected or no worm")
                return False
            
            worm_pos, _ = p.getBasePositionAndOrientation(self.worm_id)
            
            # Try multiple positions until we find a valid one
            for attempt in range(self.max_spawn_attempts):
                # Try spawning in front of the worm with some randomness
                angle = np.random.uniform(-np.pi/4, np.pi/4)  # Reduced angle range
                distance = np.random.uniform(self.food_spawn_min_dist, self.food_spawn_radius)
                
                # Bias towards spawning in front of the worm
                food_x = worm_pos[0] + distance * np.cos(angle)
                food_y = worm_pos[1] + distance * np.sin(angle)
                food_z = 0.025  # Slightly above ground
                
                # Create the food object
                try:
                    # First create visual shape
                    visual_shape = p.createVisualShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=self.food_size,
                        rgbaColor=[1, 0, 0, 1]
                    )
                    
                    # Then create collision shape
                    collision_shape = p.createCollisionShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=self.food_size
                    )
                    
                    # Finally create the body
                    food_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=collision_shape,
                        baseVisualShapeIndex=visual_shape,
                        basePosition=[food_x, food_y, food_z]
                    )
                    
                    # Only update our references if creation was successful
                    if food_id is not None:
                        self.food_visual_id = food_id
                        self.food_pos = [food_x, food_y, food_z]
                        return True
                except:
                    # If creation fails, clean up and try again
                    continue
            
            # If we failed to spawn food, use a default position
            try:
                self.food_pos = [worm_pos[0] + 0.5, worm_pos[1], 0.025]
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=self.food_size,
                    rgbaColor=[1, 0, 0, 1]
                )
                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=self.food_size
                )
                food_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=self.food_pos
                )
                if food_id is not None:
                    self.food_visual_id = food_id
                    return True
            except:
                # If even default position fails, return False
                return False
            
            return False
            
        except Exception as e:
            print(f"Error spawning food: {e}")
            return False

    def _create_stairs(self):
        """Create a staircase in the environment."""
        for i in range(self.num_stairs):
            # Calculate stair position
            x = 0.5 + i * self.stair_depth  # Start stairs 0.5 units away
            y = 0
            z = (i + 1) * self.stair_height / 2  # Half height because URDF box center is at middle
            
            # Create collision shape
            col_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[self.stair_depth/2, self.stair_width/2, (i + 1) * self.stair_height/2]
            )
            
            # Create visual shape
            vis_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[self.stair_depth/2, self.stair_width/2, (i + 1) * self.stair_height/2],
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
            
            # Create body
            stair_id = p.createMultiBody(
                baseMass=0,  # Static body
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[x, y, z]
            )
            
            # Set high friction for stairs
            p.changeDynamics(
                stair_id,
                -1,
                lateralFriction=2.0,
                spinningFriction=0.5,
                rollingFriction=0.5,
                contactStiffness=10000,
                contactDamping=100
            )
            
            self.stairs.append(stair_id)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Always ensure connection
        if not self.connected:
            self._connect_physics()
        
        # Full reset if needed
        if self.steps_since_reset >= self.max_steps or self.worm_id is None:
            try:
                p.resetSimulation()
                p.setGravity(0, 0, -9.81)
                
                # Load ground plane
                self.plane_id = p.loadURDF("plane.urdf")
                p.changeDynamics(
                    self.plane_id,
                    -1,
                    lateralFriction=2.0,
                    spinningFriction=0.5,
                    rollingFriction=0.5
                )
                
                # Create stairs
                self._create_stairs()
                
                # Load worm
                self.worm_id = p.loadURDF(self.urdf_path, [0, 0, 0.05])
                
                # Set dynamics properties
                for i in range(p.getNumJoints(self.worm_id) + 1):
                    p.changeDynamics(
                        self.worm_id,
                        i-1,
                        lateralFriction=1.5,
                        spinningFriction=0.3,
                        rollingFriction=0.3,
                        linearDamping=0.3,
                        angularDamping=0.3,
                        jointDamping=self.joint_damping,
                        contactStiffness=10000,
                        contactDamping=100
                    )
            except Exception as e:
                print(f"Error in full reset: {e}")
                self.connected = False
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        else:
            # Just reset worm position
            try:
                p.resetBasePositionAndOrientation(self.worm_id, [0, 0, 0.05], [0, 0, 0, 1])
                for joint in range(self.num_joints):
                    p.resetJointState(self.worm_id, joint, 0, 0)
            except Exception as e:
                print(f"Error in position reset: {e}")
                return self.reset(seed=seed)  # Try full reset if position reset fails
        
        # Reset state variables
        self.previous_x = 0
        self.previous_height = 0
        self.steps_since_reset = 0
        
        # Let the worm settle
        for _ in range(100):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(0.001)
        
        return self._get_observation(), {}

    def _get_closest_stair_info(self):
        """Get information about the closest stair relative to the worm."""
        worm_pos, _ = p.getBasePositionAndOrientation(self.worm_id)
        closest_dist = float('inf')
        closest_stair_pos = [0, 0, 0]
        
        for i, stair_id in enumerate(self.stairs):
            stair_pos = p.getBasePositionAndOrientation(stair_id)[0]
            dist = np.linalg.norm(np.array(worm_pos) - np.array(stair_pos))
            if dist < closest_dist:
                closest_dist = dist
                closest_stair_pos = stair_pos
        
        return closest_stair_pos

    def _handle_mouse_events(self):
        """Handle mouse clicks to place food or stairs in GUI mode."""
        if not self.render_mode:
            return
        events = p.getMouseEvents()
        for e in events:
            # Only handle mouse button down events
            if e[0] == 2:  # 2 = Mouse button event
                if e[3] == 1:
                    mouse_x, mouse_y = e[1], e[2]
                    width, height, view_mat, proj_mat, _, _ = p.getDebugVisualizerCamera()
                    ray_start, ray_end = self._compute_ray(mouse_x, mouse_y, width, height, view_mat, proj_mat)
                    hits = p.rayTest(ray_start, ray_end)
                    if hits and hits[0][0] != -1:
                        hit_obj = hits[0][0]
                        hit_pos = hits[0][3]
                        # Only allow placement on ground plane (self.plane_id)
                        if hit_obj == self.plane_id:
                            # Ignore clicks too close to the worm
                            worm_pos, _ = p.getBasePositionAndOrientation(self.worm_id)
                            dist = np.linalg.norm(np.array(hit_pos[:2]) - np.array(worm_pos[:2]))
                            if dist < 0.2:
                                continue  # Too close to worm
                            if e[4] == 0:  # Left button
                                self._place_food_at(hit_pos)
                            elif e[4] == 1:  # Right button
                                self._place_stair_at(hit_pos)

    def _compute_ray(self, mouse_x, mouse_y, width, height, view_mat, proj_mat):
        """Compute a ray from the camera through the mouse position."""
        # Convert mouse_x, mouse_y (pixels) to normalized device coordinates
        ndc_x = (2.0 * mouse_x) / width - 1.0
        ndc_y = 1.0 - (2.0 * mouse_y) / height
        # Near and far points in NDC
        ndc_near = [ndc_x, ndc_y, -1, 1]
        ndc_far = [ndc_x, ndc_y, 1, 1]
        # Inverse projection and view
        inv_proj = np.linalg.inv(np.array(proj_mat).reshape(4, 4))
        inv_view = np.linalg.inv(np.array(view_mat).reshape(4, 4))
        # Unproject
        near_world = inv_view @ (inv_proj @ np.array(ndc_near))
        far_world = inv_view @ (inv_proj @ np.array(ndc_far))
        near_world /= near_world[3]
        far_world /= far_world[3]
        return near_world[:3], far_world[:3]

    def _place_food_at(self, pos):
        """Place food at the given 3D position (on ground)."""
        if self.food_visual_id is not None:
            try:
                p.removeBody(self.food_visual_id)
            except:
                pass
            self.food_visual_id = None
        # Always place food on ground
        self.food_pos = [pos[0], pos[1], self.food_size]
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.food_size,
            rgbaColor=[1, 0, 0, 1]
        )
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.food_size
        )
        self.food_visual_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.food_pos
        )

    def _place_stair_at(self, pos):
        """Place a stair at the given 3D position (on ground)."""
        stair_height = self.stair_height
        stair_width = self.stair_width
        stair_depth = self.stair_depth
        col_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[stair_depth/2, stair_width/2, stair_height/2]
        )
        vis_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[stair_depth/2, stair_width/2, stair_height/2],
            rgbaColor=[0.8, 0.8, 0.8, 1]
        )
        stair_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[pos[0], pos[1], stair_height/2]
        )
        p.changeDynamics(
            stair_id,
            -1,
            lateralFriction=2.0,
            spinningFriction=0.5,
            rollingFriction=0.5,
            contactStiffness=10000,
            contactDamping=100
        )
        self.stairs.append(stair_id)

    def step(self, action):
        if self.render_mode:
            self._handle_mouse_events()
        
        if not self.connected:
            try:
                self._connect_physics()
                return self.reset()[0], 0.0, True, False, {}
            except:
                return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}
        
        try:
            # Apply actions and step simulation
            for _ in range(self.sim_steps_per_action):
                for joint in range(self.num_joints):
                    current_vel = p.getJointState(self.worm_id, joint)[1]
                    target_vel = action[joint] * self.max_joint_velocity
                    desired_vel = current_vel + np.clip(target_vel - current_vel, -0.05, 0.05)
                    current_pos = p.getJointState(self.worm_id, joint)[0]
                    target_pos = current_pos + desired_vel * (1/120.)
                    p.setJointMotorControl2(
                        bodyIndex=self.worm_id,
                        jointIndex=joint,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        targetVelocity=desired_vel,
                        positionGain=self.position_gain,
                        velocityGain=self.velocity_gain,
                        force=self.max_force,
                        maxVelocity=self.max_joint_velocity
                    )
                p.stepSimulation()
                if self.render_mode:
                    time.sleep(self.render_delay)
            
            # Get current state
            worm_pos, orientation = p.getBasePositionAndOrientation(self.worm_id)
            orientation_euler = p.getEulerFromQuaternion(orientation)
            
            # Calculate rewards
            height_change = worm_pos[2] - self.previous_height
            self.previous_height = worm_pos[2]
            forward_progress = worm_pos[0] - self.previous_x
            self.previous_x = worm_pos[0]
            contact_points = p.getContactPoints(self.worm_id)
            contact_reward = len(contact_points) * 0.1
            
            # Encourage forward progress much more
            forward_reward = forward_progress * 100.0  # Stronger incentive
            climbing_reward = height_change * 30.0     # Still reward climbing, but less
            
            # Reduce penalties for instability
            stability_penalty = -0.05 * (abs(orientation_euler[0]) + abs(orientation_euler[2]))
            
            # Remove velocity penalty entirely
            # Add a small base reward for any movement
            movement_reward = 0.05 if abs(forward_progress) > 1e-4 else 0.0
            
            # Combine rewards
            reward = (
                forward_reward * 0.5 +    # Strongly encourage forward movement
                climbing_reward * 0.2 +   # Still reward climbing
                stability_penalty * 0.1 + # Small penalty for instability
                contact_reward * 0.1 +    # Encourage contact
                movement_reward * 0.1     # Small reward for any movement
            )
            
            # Early termination conditions
            terminated = False
            if abs(worm_pos[1]) > 0.5:
                terminated = True
                reward = -1.0
            if abs(orientation_euler[0]) > 1.5 or abs(orientation_euler[2]) > 1.5:
                terminated = True
                reward = -1.0
            if self.render_mode and self.steps_since_reset % 100 == 0:
                print(f"\nStep {self.steps_since_reset}")
                print(f"Position: {[f'{p:.2f}' for p in worm_pos]}")
                print(f"Forward progress: {forward_progress:.3f}")
                print(f"Reward: {reward:.3f}")
            self.steps_since_reset += 1
            if self.steps_since_reset >= self.max_steps:
                terminated = True
            return self._get_observation(), reward, terminated, False, {}
        except Exception as e:
            print(f"Error in step: {e}")
            self.connected = False
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

    def _get_observation(self):
        if not self.connected:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        try:
            obs = []
            
            # Get worm state
            pos, orn = p.getBasePositionAndOrientation(self.worm_id)
            obs.extend(list(pos))
            obs.extend(list(orn))
            
            # Get joint states
            for joint in range(self.num_joints):
                joint_state = p.getJointState(self.worm_id, joint)
                obs.extend([joint_state[0], joint_state[1]])
            
            # Add food position (ensure it exists)
            if self.food_pos is None:
                self.food_pos = [2.0, 0.0, 0.025]
            obs.extend(self.food_pos)
            
            # Add closest stair information
            closest_stair_pos = self._get_closest_stair_info()
            obs.extend(closest_stair_pos)
            
            return np.array(obs, dtype=np.float32)
        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        # Clean up food object if it exists
        if self.food_visual_id is not None:
            try:
                p.removeBody(self.food_visual_id)
            except:
                pass
            self.food_visual_id = None
        
        # Disconnect from physics server
        if self.connected and self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
            self.connected = False
            self.physics_client = None 