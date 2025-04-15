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
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.num_joints + 7 + 3,),  # joints pos/vel + base pos/orient + food pos
            dtype=np.float32
        )
        
        self.previous_x = 0
        self.steps_since_reset = 0
        self.plane_id = None
        self.worm_id = None
        self.food_pos = None
        self.food_visual_id = None
        self.prev_distance_to_food = None
        self.max_steps = 2000
        
        # Updated parameters
        self.food_spawn_radius = 0.5  # Reduced radius for easier targets
        self.food_spawn_min_dist = 0.2  # Reduced minimum distance
        self.food_size = 0.05
        self.max_spawn_attempts = 50
        
        # Slower physics parameters
        self.max_joint_velocity = 1.0  # Reduced maximum velocity
        self.sim_steps_per_action = 2  # Fewer steps for smoother visualization
        
        # Updated control parameters
        self.joint_damping = 0.2  # Increased damping for slower movement
        self.position_gain = 0.3
        self.velocity_gain = 0.3
        self.max_force = 3.0  # Reduced force for gentler movement
        
        # Phase parameters for undulation
        self.target_phase_diff = np.pi / 2
        self.phase_tolerance = np.pi / 4
        
        # Visualization delay (seconds)
        self.render_delay = 0.01  # Increased delay for slower visualization

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Clean up any existing objects
        if self.food_visual_id is not None:
            try:
                p.removeBody(self.food_visual_id)
            except:
                pass
            self.food_visual_id = None
        
        # Ensure connection
        if not self.connected:
            self._connect_physics()
        
        try:
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            
            # Load ground plane with higher friction
            self.plane_id = p.loadURDF("plane.urdf")
            p.changeDynamics(self.plane_id, -1, 
                           lateralFriction=1.0,
                           spinningFriction=0.1,
                           rollingFriction=0.1)
            
            # Load worm
            self.worm_id = p.loadURDF(self.urdf_path, [0, 0, 0.1])
            
            # Set dynamics properties for all worm links
            for i in range(p.getNumJoints(self.worm_id) + 1):
                p.changeDynamics(self.worm_id, i-1,
                               lateralFriction=1.0,
                               spinningFriction=0.1,
                               rollingFriction=0.1,
                               linearDamping=0.1,
                               angularDamping=0.1,
                               jointDamping=self.joint_damping)
            
            # Initialize joints with a gentler sine wave pattern
            for joint in range(self.num_joints):
                phase = (joint / self.num_joints) * 2 * np.pi
                initial_pos = 0.1 * np.sin(phase)  # Reduced amplitude
                p.resetJointState(self.worm_id, joint, initial_pos, targetVelocity=0)
                p.setJointMotorControl2(
                    self.worm_id,
                    joint,
                    p.POSITION_CONTROL,  # Changed to position control
                    targetPosition=initial_pos,
                    targetVelocity=0,
                    positionGain=self.position_gain,
                    velocityGain=self.velocity_gain,
                    force=self.max_force
                )
            
            # Reset counters and state
            self.previous_x = 0
            self.steps_since_reset = 0
            
            # Try to spawn food and initialize distance
            if not self._spawn_food():
                # If food spawning fails, use a default position
                self.food_pos = [1.0, 0.0, 0.025]
                # Create visual marker for default food position
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
            
            worm_pos, _ = p.getBasePositionAndOrientation(self.worm_id)
            self.prev_distance_to_food = np.linalg.norm(np.array(self.food_pos) - np.array(worm_pos))
            
            # Let the worm settle
            for _ in range(100):
                p.stepSimulation()
                if self.render_mode:
                    time.sleep(0.001)
            
            return self._get_observation(), {}
            
        except Exception as e:
            print(f"Error in reset: {e}")
            self.connected = False
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        if not self.connected:
            try:
                self._connect_physics()
                return self.reset()[0], 0.0, True, False, {}
            except:
                return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}
        
        try:
            # Apply actions and step simulation
            for _ in range(self.sim_steps_per_action):
                # Apply velocity control to joints
                for joint in range(self.num_joints):
                    # Scale action to velocity with smoother transitions
                    current_vel = p.getJointState(self.worm_id, joint)[1]
                    target_vel = action[joint] * self.max_joint_velocity
                    # Smooth velocity changes
                    desired_vel = current_vel + np.clip(target_vel - current_vel, -0.1, 0.1)
                    
                    # Use velocity control
                    p.setJointMotorControl2(
                        bodyIndex=self.worm_id,
                        jointIndex=joint,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=desired_vel,
                        force=self.max_force
                    )
                
                p.stepSimulation()
                if self.render_mode:
                    time.sleep(self.render_delay)  # Consistent delay for visualization
            
            # Get current state
            worm_pos, orientation = p.getBasePositionAndOrientation(self.worm_id)
            
            # Calculate rewards
            current_distance = np.linalg.norm(np.array(self.food_pos) - np.array(worm_pos))
            distance_reward = (self.prev_distance_to_food - current_distance) * 20.0  # Increased weight
            self.prev_distance_to_food = current_distance
            
            # Calculate forward progress
            forward_progress = worm_pos[0] - self.previous_x
            self.previous_x = worm_pos[0]
            progress_reward = forward_progress * 50.0  # Increased weight
            
            # Softer velocity penalty
            velocity_penalty = 0
            for joint in range(self.num_joints):
                joint_vel = p.getJointState(self.worm_id, joint)[1]
                velocity_penalty -= 0.0001 * (joint_vel ** 2)  # Reduced penalty
            
            # Enhanced undulation reward
            undulation_reward = 0.0
            joint_positions = [p.getJointState(self.worm_id, i)[0] for i in range(self.num_joints)]
            
            # Calculate phase differences between adjacent joints
            for i in range(self.num_joints - 1):
                phase_diff = abs(joint_positions[i] - joint_positions[i + 1])
                # Reward proper phase relationships
                if abs(phase_diff - self.target_phase_diff) < self.phase_tolerance:
                    undulation_reward += 1.0  # Increased reward
            
            # Encourage alternating joint movements
            for i in range(self.num_joints - 2):
                if (joint_positions[i] * joint_positions[i + 2]) < 0:  # Opposite signs
                    undulation_reward += 0.5
            
            # Height reward
            height_reward = 0.0
            if 0.01 < worm_pos[2] < 0.1:  # Tighter height range
                height_reward = 0.5
            elif worm_pos[2] >= 0.1:
                height_reward = -0.1 * (worm_pos[2] - 0.1)
            
            # Orientation reward
            orientation_euler = p.getEulerFromQuaternion(orientation)
            orientation_penalty = -0.1 * (abs(orientation_euler[0]) + abs(orientation_euler[2]))
            
            # Base reward for staying alive
            survival_reward = 0.1
            
            # Combine all rewards with updated weights
            reward = (
                distance_reward * 0.3 +      # Distance to food
                progress_reward * 0.3 +      # Forward progress
                velocity_penalty * 0.05 +    # Velocity penalty (reduced weight)
                height_reward * 0.1 +        # Height maintenance
                survival_reward * 0.05 +     # Survival
                undulation_reward * 0.15 +   # Undulation pattern
                orientation_penalty * 0.05    # Orientation
            )
            
            # Food collection bonus
            if current_distance < self.food_size * 2:
                reward += 20.0  # Increased food reward
                if not self._spawn_food():
                    reward -= 1.0  # Reduced penalty
            
            self.steps_since_reset += 1
            
            # Updated termination conditions
            terminated = False
            
            if worm_pos[2] < 0.005 or worm_pos[2] > 0.2:  # Updated height limits
                terminated = True
                reward = -1.0  # Reduced penalty
            
            if abs(orientation_euler[0]) > 1.0 or abs(orientation_euler[2]) > 1.0:  # More strict angles
                terminated = True
                reward = -1.0
            
            if self.steps_since_reset >= self.max_steps:
                terminated = True
            
            # Debug info
            if self.render_mode and self.steps_since_reset % 100 == 0:
                print(f"\nStep {self.steps_since_reset}/{self.max_steps}")
                print(f"Worm position: {[f'{p:.2f}' for p in worm_pos]}")
                print(f"Distance to food: {current_distance:.2f}")
                print(f"Forward progress: {forward_progress:.3f}")
                print(f"Undulation reward: {undulation_reward:.2f}")
                print(f"Total reward: {reward:.2f}")
            
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
            
            pos, orn = p.getBasePositionAndOrientation(self.worm_id)
            obs.extend(list(pos))
            obs.extend(list(orn))
            
            for joint in range(self.num_joints):
                joint_state = p.getJointState(self.worm_id, joint)
                obs.extend([joint_state[0], joint_state[1]])
            
            obs.extend(self.food_pos)
            
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