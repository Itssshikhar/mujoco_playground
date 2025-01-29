"""Domain randomization for the ZBot environment."""
import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor friction: =U(0.3, 1.2) - Increased range
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.3, maxval=1.2)
    )

    # Scale static friction: *U(0.8, 1.2)
    rng, key = jax.random.split(rng)
    num_dofs = model.dof_frictionloss[6:].shape[0]
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(num_dofs,), minval=0.8, maxval=1.2
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(0.95, 1.1)
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(num_dofs,), minval=0.95, maxval=1.1
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale all link masses: *U(0.8, 1.2)
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.8, maxval=1.2
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-2.0, 2.0)
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-2.0, maxval=2.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Jitter qpos0: +U(-0.1, 0.1)
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(num_dofs,), minval=-0.1, maxval=0.1)
    )

    # Add damping randomization: *U(0.9, 1.1)
    rng, key = jax.random.split(rng)
    damping = model.dof_damping[6:] * jax.random.uniform(
        key, shape=(num_dofs,), minval=0.9, maxval=1.1
    )
    dof_damping = model.dof_damping.at[6:].set(damping)

    # Add stiffness randomization: *U(0.9, 1.1)
    rng, key = jax.random.split(rng)
    num_joints = model.jnt_stiffness[6:].shape[0]  # Get actual number of joints
    stiffness = model.jnt_stiffness[6:] * jax.random.uniform(
        key, shape=(num_joints,), minval=0.9, maxval=1.1
    )
    jnt_stiffness = model.jnt_stiffness.at[6:].set(stiffness)

    # Add actuator gear randomization: *U(0.95, 1.05)
    rng, key = jax.random.split(rng)
    gear = model.actuator_gear[:, 0] * jax.random.uniform(
        key, shape=(model.nu,), minval=0.95, maxval=1.05
    )
    actuator_gear = model.actuator_gear.at[:, 0].set(gear)

    return (
        geom_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
        dof_damping,
        jnt_stiffness,
        actuator_gear,
    )

  (
      friction,
      frictionloss,
      armature,
      body_mass,
      qpos0,
      damping,
      stiffness,
      gear,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
      "dof_damping": 0,
      "jnt_stiffness": 0,
      "actuator_gear": 0,
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_damping": damping,
      "jnt_stiffness": stiffness,
      "actuator_gear": gear,
  })

  return model, in_axes
