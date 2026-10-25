//! Trees: seedlings turn woody and grow, trees bear fruit that hangs, falls and rots, seeds ride
//! the wind, and trunks are solid in every physics world.

use evolve_sim::body::Tag;
use evolve_sim::brain::output;
use evolve_sim::forest::{Fruit, TRUNK_HEIGHT};
use evolve_sim::genome::Genome;
use evolve_sim::math::Vec3;
use evolve_sim::rng::Rng;
use evolve_sim::sim::{Food, PlantKind, TICKS_PER_SECOND};
use evolve_sim::{Sim, SimParams, save};
use half::f16;
use rapier3d::prelude::{QueryFilter, Ray};

/// A world with no plants or trees of its own, so each test puts down only what it needs.
fn bare_params() -> SimParams {
    SimParams {
        min_population: 0,
        stones: 0,
        terrain_resolution: 65,
        weather: false,
        plant_max_count: 0,
        plant_spawn_rate: 0.0,
        plant_spread_rate: 0.0,
        tree_max_count: 0,
        ..SimParams::default()
    }
}

fn run(sim: &mut Sim, secs: f32) {
    for _ in 0..(secs * TICKS_PER_SECOND as f32) as usize {
        sim.step();
    }
}

/// A limbless block that always wants to eat.
fn eater() -> Genome {
    let mut g = Genome::random(&mut Rng::new(13), &SimParams::default());
    g.spine.segments = 0;
    for l in &mut g.limbs {
        l.present = false;
    }
    let h = g.brain.h;
    g.brain.w_out.0[output::EAT * h..(output::EAT + 1) * h].fill(f16::ZERO);
    g.brain.b_out[output::EAT] = 5.0;
    g
}

/// A hungry eater set down and settled; returns its id and where its mouth is.
fn hungry_eater(sim: &mut Sim) -> (u64, Vec3) {
    let at = sim.random_land_point(30.0);
    let id = sim.spawn_creature(eater(), at, 0.3, 10.0, 0, None);
    run(sim, 1.5);
    (id, mouth(sim, id))
}

fn mouth(sim: &Sim, id: u64) -> Vec3 {
    let c = sim.creature(id).unwrap();
    sim.torso_of(c).unwrap().position().transform_point(Vec3::new(c.torso_half().x, 0.0, 0.0))
}

/// A fruit hanging on tree `tree` at `pos`, or lying there if `tree` is None.
fn put_fruit(sim: &mut Sim, pos: Vec3, tree: Option<u64>) -> u64 {
    let id = sim.next_id();
    let energy = sim.params.fruit_energy;
    sim.fruit.push(Fruit { id, pos, energy, age: 0.0, tree });
    id
}

fn fruit_energy(sim: &Sim, id: u64) -> f32 {
    sim.fruit.iter().find(|f| f.id == id).map_or(0.0, |f| f.energy)
}

#[test]
fn a_seedling_left_alone_turns_woody_grows_and_bears_fruit() {
    let mut sim = Sim::new(SimParams {
        plant_max_count: 10,
        tree_max_count: 1,
        tree_height: 6.0,
        tree_growth_time: 60.0,
        fruit_rate: 30.0,
        ..bare_params()
    });
    let at = sim.random_land_point(20.0);
    sim.plants.clear();
    sim.add_plant_of(at, sim.params.plant_max_energy, PlantKind::Seedling).unwrap();
    run(&mut sim, 1.5);
    assert_eq!(sim.trees.len(), 1, "a full-grown seedling turns into a tree");
    assert!(sim.plants.iter().all(|f| f.kind != PlantKind::Seedling), "and is no longer a soft plant");
    let id = sim.trees[0].id;
    let young = sim.trees[0].height;
    run(&mut sim, 150.0);
    let t = sim.trees.iter().find(|t| t.id == id).expect("it lives");
    assert!(t.height > 2.0 * young && t.height > TRUNK_HEIGHT, "it grows: {young:.1} -> {:.1} m", t.height);
    assert!(t.wood > 0.0);
    assert!(sim.shards.iter().all(|s| s.trunks.contains_key(&id)), "its trunk is solid in every physics world");
    let hanging: Vec<&Fruit> = sim.fruit.iter().filter(|f| f.tree == Some(id)).collect();
    assert!(!hanging.is_empty(), "a grown tree bears fruit");
    // (Each where the crown was when it formed: this tree grows a hundred times faster than
    // usual.)
    for f in hanging {
        let ground = sim.terrain.height_at(f.pos.x, f.pos.z);
        let out = (f.pos.x - t.pos.x).hypot(f.pos.z - t.pos.z);
        assert!(f.pos.y > ground + 1.0 && out < t.crown_radius(), "fruit hangs in the crown: {:.1} m up and {out:.1} m out", f.pos.y - ground);
    }
}

#[test]
fn wood_is_hard_to_eat_and_fruit_is_easy() {
    let mut sim = Sim::new(SimParams { tree_max_count: 4, ..bare_params() });
    let (_, m) = hungry_eater(&mut sim);
    // A fallen fruit at its mouth.
    let ground = sim.terrain.height_at(m.x, m.z);
    let fruit = put_fruit(&mut sim, Vec3::new(m.x, ground, m.z), None);
    let before = fruit_energy(&sim, fruit);
    run(&mut sim, 1.0);
    let ate_fruit = before - fruit_energy(&sim, fruit);
    sim.fruit.clear();
    assert!(ate_fruit > 1.0, "it bites the fruit: {ate_fruit:.2}");

    // A young woody tree (no solid trunk yet, so it doesn't shove the eater) at its mouth.
    let (id, m) = (sim.creatures[0].id, mouth(&sim, sim.creatures[0].id));
    let k = sim.add_tree(m).unwrap();
    sim.trees[k].wood = 100.0;
    let tree = sim.trees[k].id;
    run(&mut sim, 1.0);
    let wood = sim.trees.iter().find(|t| t.id == tree).unwrap().wood;
    let gnawed = 100.0 - wood;
    assert!(gnawed > 0.0, "with nothing else in reach it gnaws the wood");
    assert!(gnawed < 0.2 * ate_fruit, "but gets far less from it: {gnawed:.2} of wood vs {ate_fruit:.2} of fruit");
    assert!(sim.creature(id).is_some());
}

#[test]
fn hanging_fruit_feeds_only_a_mouth_that_reaches_it() {
    let mut sim = Sim::new(SimParams { tree_max_count: 4, ..bare_params() });
    let (_, m) = hungry_eater(&mut sim);
    // The tree stands well away; its fruit hangs over the eater.
    let k = sim.add_tree(m + Vec3::new(15.0, 0.0, 0.0)).unwrap();
    let tree = sim.trees[k].id;
    let high = put_fruit(&mut sim, m + Vec3::Y * 3.0, Some(tree));
    run(&mut sim, 1.0);
    assert_eq!(fruit_energy(&sim, high), sim.params.fruit_energy, "out of reach overhead");
    sim.fruit.clear();
    let low = put_fruit(&mut sim, m + Vec3::Y * 0.2, Some(tree));
    run(&mut sim, 1.0);
    assert!(fruit_energy(&sim, low) < sim.params.fruit_energy, "eaten once it hangs within reach");
}

#[test]
fn overripe_fruit_falls_rots_and_sprouts() {
    let mut sim = Sim::new(SimParams {
        plant_max_count: 100,
        tree_max_count: 4,
        fruit_hang_time: 3.0,
        fruit_rot_time: 5.0,
        fruit_sprout: 1.0,
        ..bare_params()
    });
    sim.plants.clear();
    let at = sim.random_land_point(30.0);
    let k = sim.add_tree(at).unwrap();
    let tree = sim.trees[k].id;
    // Fruit hanging 3 m up, some way off the trunk (out of its shade), at the best ground about.
    let mut spots = Vec::new();
    let mut rng = Rng::new(3);
    while spots.len() < 8 {
        let p = at + Vec3::new(rng.range(-12.0, 12.0), 0.0, rng.range(-12.0, 12.0));
        let s = sim.terrain.plant_suitability(p.x, p.z, sim.params.water_level, sim.params.terrain_height);
        if s > 0.3 && (p - at).length() > 5.0 {
            spots.push(Vec3::new(p.x, sim.terrain.height_at(p.x, p.z) + 3.0, p.z));
        }
    }
    let ids: Vec<u64> = spots.iter().map(|&p| put_fruit(&mut sim, p, Some(tree))).collect();
    run(&mut sim, 4.5);
    for &id in &ids {
        let f = sim.fruit.iter().find(|f| f.id == id).expect("still there");
        assert!(f.tree.is_none(), "overripe fruit falls");
        assert!((f.pos.y - sim.terrain.height_at(f.pos.x, f.pos.z)).abs() < 0.05, "to the ground");
    }
    run(&mut sim, 6.0);
    assert!(ids.iter().all(|&id| sim.fruit.iter().all(|f| f.id != id)), "and rots away");
    let sprouted = spots
        .iter()
        .filter(|s| sim.plants.iter().any(|p| p.kind == PlantKind::Seedling && (p.pos.x - s.x).hypot(p.pos.z - s.z) < 0.1))
        .count();
    assert!(sprouted > 0, "fruit left to rot can sprout a tree seedling");
}

#[test]
fn seeds_drift_downwind() {
    let mut sim = Sim::new(bare_params());
    let from = sim.random_land_point(20.0);
    let mean = |sim: &mut Sim, wind: [f32; 2]| {
        sim.weather.wind = wind;
        let n = 400;
        let sum = (0..n).fold(Vec3::ZERO, |acc, _| acc + (sim.seed_landing(from, 7.0) - from));
        sum / n as f32
    };
    let calm = mean(&mut sim, [0.0, 0.0]);
    assert!(calm.x.hypot(calm.z) < 0.5, "in a calm they land around the tree: {calm:?}");
    let breeze = mean(&mut sim, [3.0, 0.0]);
    let fall = 7.0 / sim.params.seed_fall_speed;
    assert!((breeze.x - 3.0 * fall).abs() < 0.15 * 3.0 * fall, "carried downwind for as long as they fall: {breeze:?}");
    assert!(breeze.z.abs() < 0.5, "and not across it: {breeze:?}");
    let storm = mean(&mut sim, [0.0, -15.0]);
    assert!(storm.z < -4.0 * breeze.x, "a storm carries them much further: {storm:?}");
}

#[test]
fn a_trunk_stands_in_every_physics_world_and_holds_up_what_lands_on_it() {
    let mut sim = Sim::new(SimParams { tree_max_count: 4, physics_shards: 4, ..bare_params() });
    let at = sim.random_land_point(30.0);
    let k = sim.add_tree(at).unwrap();
    sim.trees[k].height = 8.0;
    let (id, base) = (sim.trees[k].id, sim.trees[k].pos);
    sim.sync_trunks();
    // (Rays find colliders as of the last step.)
    sim.step();
    for (s, shard) in sim.shards.iter().enumerate() {
        assert!(shard.trunks.contains_key(&id), "world {s} has the trunk");
        let qp = shard.world.query_pipeline_with_filter(QueryFilter::default());
        let from = base + Vec3::new(-3.0, 2.0, 0.0);
        let hit = qp.cast_ray(&Ray::new(from, Vec3::X), 6.0, true).map(|(h, toi)| (Tag::decode(shard.world.colliders[h].user_data), toi));
        assert!(matches!(hit, Some((Tag::Tree { id: t }, toi)) if t == id && toi < 3.0), "a ray in world {s} hits it: {hit:?}");
    }
    // A stone dropped on it comes to rest on top, about 4.8 m up, instead of on the ground.
    let top = base.y + 0.6 * 8.0;
    let stone = sim.add_obstacle(base + Vec3::new(5.0, 0.0, 0.0), evolve_sim::sim::ObstacleShape::Ball { radius: 0.25 }, true, 0.0);
    let o = sim.obstacles.iter().find(|o| o.id == stone).unwrap();
    let (shard, body) = (o.shard, o.body);
    sim.shards[shard].world.bodies[body].set_translation(Vec3::new(base.x, top + 1.0, base.z), true);
    run(&mut sim, 0.6);
    let o = sim.obstacles.iter().find(|o| o.id == stone).unwrap();
    let y = sim.shards[o.shard].world.bodies[o.body].translation().y;
    assert!(y > top - 0.5, "the trunk holds it up: {:.1} m above the ground", y - base.y);

    // Once the tree has fallen its trunk is gone from every world.
    sim.trees[k].fallen = Some(0.0);
    sim.sync_trunks();
    assert!(sim.shards.iter().all(|s| !s.trunks.contains_key(&id)));
}

#[test]
fn a_forest_saves_and_loads() {
    let mut sim = Sim::new(SimParams { tree_max_count: 40, plant_max_count: 200, ..bare_params() });
    run(&mut sim, 2.0);
    let k = sim.trees.iter().position(|t| t.solid()).expect("a new world starts with grown trees");
    let tree = sim.trees[k].id;
    let crown = sim.trees[k].pos + Vec3::Y * 4.0;
    put_fruit(&mut sim, crown, Some(tree));
    let dir = std::env::temp_dir().join(format!("evolve-forest-{}", std::process::id()));
    let path = dir.join("forest.evo");
    save::save(&mut sim, &path).unwrap();
    let loaded = save::load(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(loaded.trees.len(), sim.trees.len());
    assert_eq!(loaded.fruit.len(), sim.fruit.len());
    assert!(loaded.shards.iter().all(|s| s.trunks.contains_key(&tree)), "trunks come back in every world");
}

#[test]
fn plants_saved_before_trees_load_as_grass() {
    #[derive(serde::Serialize)]
    struct OldFood {
        id: u64,
        pos: Vec3,
        energy: f32,
    }
    let bytes = rmp_serde::encode::to_vec_named(&OldFood { id: 7, pos: Vec3::new(1.0, 2.0, 3.0), energy: 12.0 }).unwrap();
    let f: Food = rmp_serde::decode::from_slice(&bytes).unwrap();
    assert_eq!((f.id, f.energy, f.kind), (7, 12.0, PlantKind::Grass));
}
