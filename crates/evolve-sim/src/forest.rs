//! Trees. A tree seedling is a soft plant like grass, so grazing can keep a tree from ever
//! growing. One that grows full uneaten turns woody: a sapling that grows, over about an hour on
//! good ground, into a tree 8-12 m tall with a solid trunk. Wood is hard to eat (`wood_bite`),
//! but grown trees bear fruit: it hangs in the crown, where only a mouth that reaches it can eat
//! it, falls when overripe, and rots where it lies, sometimes sprouting a seedling. Trees also
//! send seeds off on the wind, and old trees die and fall.

use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

use crate::body::Tag;
use crate::math::Vec3;
use crate::sim::{PlantKind, Sim};

/// Height of a seedling the moment it turns woody, m.
pub const SAPLING_HEIGHT: f32 = 1.0;
/// From this height a tree's trunk is solid, m.
pub const TRUNK_HEIGHT: f32 = 2.0;
/// How long a fallen tree lies before it's gone, s.
pub const FALLEN_FOR: f32 = 60.0;
/// Most fruit a full-grown tree carries at once (fewer while it's small).
const FRUIT_CAP: f32 = 12.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tree {
    pub id: u64,
    /// The foot of the trunk, on the ground.
    pub pos: Vec3,
    pub height: f32,
    /// The height it grows toward (`tree_height`, give or take a fifth).
    pub max_height: f32,
    /// Energy in its wood: what gnawing takes. A tree gnawed below half the wood its size
    /// should have stops growing and bearing until it heals; at none it dies.
    pub wood: f32,
    pub age: f32,
    pub lifespan: f32,
    /// Progress toward its next fruit, 0..1.
    pub fruiting: f32,
    /// Seconds since it died and fell; None while it stands.
    pub fallen: Option<f32>,
    /// The height its trunk collider was last sized for (0 while it has none).
    #[serde(default)]
    pub trunk: f32,
}

impl Tree {
    pub fn standing(&self) -> bool {
        self.fallen.is_none()
    }

    /// How grown it is, 0..1.
    pub fn grown(&self) -> f32 {
        (self.height / self.max_height.max(1e-3)).clamp(0.0, 1.0)
    }

    pub fn trunk_radius(&self) -> f32 {
        0.04 * self.height
    }

    pub fn crown_radius(&self) -> f32 {
        0.35 * self.height
    }

    /// Whether its trunk is solid.
    pub fn solid(&self) -> bool {
        self.standing() && self.height >= TRUNK_HEIGHT
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fruit {
    pub id: u64,
    pub pos: Vec3,
    pub energy: f32,
    /// Seconds since it formed, while hanging; since it fell, on the ground.
    pub age: f32,
    /// The tree it hangs on; None once it has fallen.
    pub tree: Option<u64>,
}

impl Fruit {
    pub fn hanging(&self) -> bool {
        self.tree.is_some()
    }
}

/// The wood a tree of `height` holds when it's whole.
fn wood_for(p: &crate::params::SimParams, height: f32) -> f32 {
    p.tree_wood * (height / p.tree_height.max(1e-3)).powi(2)
}

impl Sim {
    /// A new world starts as a forest: half the trees it can hold, of every age.
    pub(crate) fn plant_forest(&mut self) {
        let target = self.params.tree_max_count as usize / 2;
        let mut tries = 0;
        while self.trees.len() < target && tries < target * 30 {
            tries += 1;
            let half = self.terrain.half() - 2.0;
            let pos = Vec3::new(self.rng.range(-half, half), 0.0, self.rng.range(-half, half));
            let s = self.suitability(pos);
            if s <= 0.0 || !self.rng.chance(s.sqrt()) || self.near_tree(pos, self.params.tree_min_spacing) {
                continue;
            }
            let grown = self.rng.range(0.1, 1.0);
            if let Some(i) = self.add_tree(pos) {
                let t = &mut self.trees[i];
                t.height = (t.max_height * grown).max(SAPLING_HEIGHT);
                t.wood = wood_for(&self.params, t.height);
                t.age = t.lifespan * grown * 0.5;
            }
        }
        self.sync_trunks();
    }

    fn suitability(&self, pos: Vec3) -> f32 {
        if !self.terrain.contains(pos.x, pos.z, 1.0) {
            return 0.0;
        }
        self.terrain.plant_suitability(pos.x, pos.z, self.params.water_level, self.params.terrain_height)
    }

    /// Whether a standing tree's trunk is within `r` of `pos` (horizontally).
    pub(crate) fn near_tree(&self, pos: Vec3, r: f32) -> bool {
        self.trees.iter().filter(|t| t.standing()).any(|t| {
            let (dx, dz) = (t.pos.x - pos.x, t.pos.z - pos.z);
            dx * dx + dz * dz < r * r
        })
    }

    /// A sapling at `pos`, unless the world holds all the trees it can. Returns its index.
    pub fn add_tree(&mut self, pos: Vec3) -> Option<usize> {
        if self.trees.len() as u32 >= self.params.tree_max_count || !self.terrain.contains(pos.x, pos.z, 1.0) {
            return None;
        }
        let id = self.next_id();
        let p = &self.params;
        let max_height = p.tree_height * self.rng.range(0.8, 1.2);
        let lifespan = p.tree_lifespan * self.rng.range(0.7, 1.3);
        let wood = wood_for(&self.params, SAPLING_HEIGHT);
        let y = self.terrain.height_at(pos.x, pos.z);
        self.trees.push(Tree {
            id,
            pos: Vec3::new(pos.x, y, pos.z),
            height: SAPLING_HEIGHT,
            max_height,
            wood,
            age: 0.0,
            lifespan,
            fruiting: 0.0,
            fallen: None,
            trunk: 0.0,
        });
        Some(self.trees.len() - 1)
    }

    /// Where a seed let go at `from` comes down: carried downwind for as long as it takes to fall
    /// from `height` above the ground (`seed_fall_speed`), give or take a gust.
    pub fn seed_landing(&mut self, from: Vec3, height: f32) -> Vec3 {
        let [wx, wz] = self.weather.wind;
        let fall = height.max(0.5) / self.params.seed_fall_speed.max(0.05);
        let gust = self.rng.range(0.5, 1.5);
        let a = self.rng.range(0.0, std::f32::consts::TAU);
        let r = self.rng.range(0.0, 2.0);
        Vec3::new(from.x + wx * fall * gust + a.cos() * r, 0.0, from.z + wz * fall * gust + a.sin() * r)
    }

    /// Once per simulated second: trees grow, bear fruit and seed; fruit ripens, falls and rots;
    /// full-grown seedlings turn woody; old and gnawed-down trees fall.
    pub(crate) fn grow_forest(&mut self, dt: f32) {
        let p = self.params.clone();
        let rain = 1.0 + p.rain_plant_growth * self.weather.rain;
        let rate = 3.0 / p.tree_growth_time.max(1.0);
        let mut hanging: std::collections::HashMap<u64, u32> = Default::default();
        for f in self.fruit.iter().filter(|f| f.hanging()) {
            *hanging.entry(f.tree.unwrap_or(0)).or_default() += 1;
        }
        let mut new_fruit = Vec::new();
        let mut seeds = Vec::new();
        let (terrain, rng) = (&self.terrain, &mut self.rng);
        for t in &mut self.trees {
            if let Some(f) = &mut t.fallen {
                *f += dt;
                continue;
            }
            t.age += dt;
            let s = terrain.plant_suitability(t.pos.x, t.pos.z, p.water_level, p.terrain_height);
            let whole = wood_for(&p, t.height);
            if s <= 0.0 {
                // Flooded or on bare rock: it dies back over a few minutes.
                t.wood -= whole * 0.005 * dt;
            } else if t.wood < 0.5 * whole {
                // Gnawed: it heals before it grows or bears again.
                t.wood += whole * 0.002 * dt;
            } else {
                let grow = (0.3 + 0.7 * s) * rain;
                let before = t.height;
                t.height += (t.max_height - t.height) * rate * grow * dt;
                t.wood += wood_for(&p, t.height) - wood_for(&p, before);
                let g = t.grown();
                if g >= 0.5 {
                    t.fruiting += p.fruit_rate / 60.0 * g * g * grow * dt;
                    while t.fruiting >= 1.0 {
                        t.fruiting -= 1.0;
                        let n = hanging.entry(t.id).or_default();
                        if (*n as f32) < (FRUIT_CAP * g * g).ceil() {
                            *n += 1;
                            // Somewhere in the crown, from a third of the way up to most of it.
                            let a = rng.range(0.0, std::f32::consts::TAU);
                            let r = t.crown_radius() * rng.range(0.15, 0.85).sqrt();
                            let y = t.height * rng.range(0.35, 0.8);
                            new_fruit.push((t.id, t.pos + Vec3::new(a.cos() * r, y, a.sin() * r)));
                        }
                    }
                    if rng.chance(p.tree_seed_rate / 60.0 * g * g * dt) {
                        seeds.push((t.pos, t.height * 0.7));
                    }
                }
            }
            if t.age >= t.lifespan || t.wood <= 0.0 {
                t.fallen = Some(0.0);
            }
        }
        self.trees.retain(|t| t.fallen.is_none_or(|f| f < FALLEN_FOR));
        for (tree, pos) in new_fruit {
            let id = self.next_id();
            self.fruit.push(Fruit { id, pos, energy: p.fruit_energy, age: 0.0, tree: Some(tree) });
        }

        // Fruit: hangs until overripe (or its tree falls), then lies where it drops and rots.
        let standing: std::collections::HashSet<u64> = self.trees.iter().filter(|t| t.standing()).map(|t| t.id).collect();
        let rot = p.fruit_energy / p.fruit_rot_time.max(1.0);
        let terrain = &self.terrain;
        for f in &mut self.fruit {
            f.age += dt;
            match f.tree {
                Some(t) if f.age < p.fruit_hang_time && standing.contains(&t) => {}
                Some(_) => {
                    f.tree = None;
                    f.age = 0.0;
                    f.pos.y = terrain.height_at(f.pos.x, f.pos.z);
                }
                None => f.energy -= rot * dt,
            }
        }
        let rotted: Vec<Vec3> = self.fruit.iter().filter(|f| !f.hanging() && f.energy <= 0.05).map(|f| f.pos).collect();
        self.fruit.retain(|f| f.hanging() || f.energy > 0.05);

        // New seedlings: from seeds on the wind, and from fruit that rotted where it fell.
        let mut grid = self.plant_grid();
        for (from, height) in seeds {
            let at = self.seed_landing(from, height);
            self.try_plant_at(at, 1.0, PlantKind::Seedling, &mut grid);
        }
        for at in rotted {
            if self.rng.chance(p.fruit_sprout) {
                self.try_plant_at(at, 1.0, PlantKind::Seedling, &mut grid);
            }
        }

        // Seedlings that grew full uneaten turn woody, where there's room for a tree.
        let mut k = 0;
        while k < self.plants.len() {
            let f = &self.plants[k];
            if f.kind == PlantKind::Seedling
                && f.energy >= p.plant_max_energy
                && (self.trees.len() as u32) < p.tree_max_count
                && !self.near_tree(f.pos, p.tree_min_spacing)
            {
                let pos = f.pos;
                self.plants.swap_remove(k);
                self.add_tree(pos);
            } else {
                k += 1;
            }
        }
        self.sync_trunks();
    }

    /// Puts trees and fallen fruit back on the ground after it changed (everywhere, or within
    /// `area` = (x0, x1, z0, z1)).
    pub(crate) fn reseat_trees(&mut self, area: Option<(f32, f32, f32, f32)>) {
        let inside = |pos: Vec3| area.is_none_or(|(x0, x1, z0, z1)| pos.x >= x0 && pos.x <= x1 && pos.z >= z0 && pos.z <= z1);
        let terrain = &self.terrain;
        self.trees.retain(|t| terrain.contains(t.pos.x, t.pos.z, 0.5));
        for t in self.trees.iter_mut().filter(|t| inside(t.pos)) {
            let y = terrain.height_at(t.pos.x, t.pos.z);
            if (y - t.pos.y).abs() > 0.01 {
                t.pos.y = y;
                // Moves its trunk too.
                t.trunk = 0.0;
            }
        }
        self.fruit.retain(|f| terrain.contains(f.pos.x, f.pos.z, 0.5));
        for f in self.fruit.iter_mut().filter(|f| !f.hanging() && inside(f.pos)) {
            f.pos.y = terrain.height_at(f.pos.x, f.pos.z);
        }
        self.sync_trunks();
    }

    /// Removes a tree; its fruit falls.
    pub fn remove_tree(&mut self, id: u64) {
        self.trees.retain(|t| t.id != id);
        for f in self.fruit.iter_mut().filter(|f| f.tree == Some(id)) {
            f.tree = None;
            f.age = 0.0;
        }
        self.reseat_fruit();
        self.sync_trunks();
    }

    fn reseat_fruit(&mut self) {
        let terrain = &self.terrain;
        for f in self.fruit.iter_mut().filter(|f| !f.hanging()) {
            f.pos.y = terrain.height_at(f.pos.x, f.pos.z);
        }
    }

    /// Gives every physics world a trunk for each solid tree, sized to it, and takes away the
    /// trunks of trees gone, fallen or not yet solid. Trunks are resized only when a tree has
    /// grown by a twentieth since, which costs little.
    pub fn sync_trunks(&mut self) {
        let friction = self.params.ground_friction;
        for t in &mut self.trees {
            let want = t.solid();
            let resize = want && (t.trunk <= 0.0 || (t.height - t.trunk).abs() > 0.05 * t.height);
            // Sunk a little into the ground, so slopes leave no gap under it; up to where the
            // crown begins.
            let sink = 0.3;
            let top = 0.6 * t.height;
            let half = 0.5 * (top + sink);
            let at = t.pos + Vec3::Y * (top - half);
            let r = t.trunk_radius().max(0.08);
            for shard in &mut self.shards {
                match (want, shard.trunks.get(&t.id).copied()) {
                    (true, None) => {
                        let c = ColliderBuilder::cylinder(half, r)
                            .translation(at)
                            .friction(friction)
                            .user_data(Tag::Tree { id: t.id }.encode())
                            .build();
                        let h = shard.world.colliders.insert(c);
                        shard.trunks.insert(t.id, h);
                    }
                    (true, Some(h)) if resize => {
                        if let Some(c) = shard.world.colliders.get_mut(h) {
                            c.set_shape(SharedShape::cylinder(half, r));
                            c.set_translation(at);
                        }
                    }
                    (false, Some(h)) => {
                        shard.world.remove_collider(h);
                        shard.trunks.remove(&t.id);
                    }
                    _ => {}
                }
            }
            t.trunk = if want { if resize { t.height } else { t.trunk } } else { 0.0 };
        }
        // Trees that are gone altogether.
        let ids: std::collections::HashSet<u64> = self.trees.iter().map(|t| t.id).collect();
        for shard in &mut self.shards {
            let stale: Vec<u64> = shard.trunks.keys().filter(|id| !ids.contains(id)).copied().collect();
            for id in stale {
                if let Some(h) = shard.trunks.remove(&id) {
                    shard.world.remove_collider(h);
                }
            }
        }
    }
}
