//! Things the user can do to a running world.

use crate::genome::Genome;
use crate::math::Vec3;
use crate::params::SimParams;
use crate::sim::{DeathCause, Grab, ObstacleShape, Sim};

#[derive(Clone, Debug)]
pub enum Command {
    SetParams(Box<SimParams>),
    RegenerateTerrain,
    SpawnPlant { pos: [f32; 3] },
    SpawnPlants { center: [f32; 3], radius: f32, count: u32 },
    SpawnMeat { pos: [f32; 3] },
    SpawnObstacle { pos: [f32; 3], shape: ObstacleShape, dynamic: bool, yaw: f32 },
    /// Drop a creature: random/gene pool if `from` is None, else a mutated copy of `from`.
    SpawnCreature { pos: Option<[f32; 3]>, from: Option<u64>, count: u32 },
    /// Drop mutated copies of a genome (from the zoo) at `pos` or a random spot.
    SpawnGenome { genome: Box<Genome>, generation: u32, pos: Option<[f32; 3]>, count: u32 },
    RemoveCreature(u64),
    KillCreature(u64),
    FeedCreature(u64),
    RemovePlant(u64),
    RemoveMeat(u64),
    RemoveObstacle(u64),
    Select(Option<u64>),
    /// Grab a creature part (`creature` Some) or a dynamic obstacle by its body, at a local point.
    GrabCreature { creature: u64, part: usize, local: [f32; 3], target: [f32; 3] },
    GrabObstacle { obstacle: u64, local: [f32; 3], target: [f32; 3] },
    DragTo([f32; 3]),
    Release,
    ClearCreatures,
    ClearPlants,
    ClearObstacles,
}

impl Sim {
    pub fn apply(&mut self, cmd: Command) {
        match cmd {
            Command::SetParams(p) => self.set_params(*p),
            Command::RegenerateTerrain => self.regenerate_terrain(),
            Command::SpawnPlant { pos } => {
                let e = self.params.plant_max_energy * 0.5;
                self.add_plant(Vec3::from(pos), e);
            }
            Command::SpawnPlants { center, radius, count } => {
                let e = self.params.plant_max_energy * 0.5;
                let c = Vec3::from(center);
                for _ in 0..count {
                    let a = self.rng.range(0.0, std::f32::consts::TAU);
                    let d = radius * self.rng.f32().sqrt();
                    self.add_plant(c + Vec3::new(a.cos() * d, 0.0, a.sin() * d), e);
                }
            }
            Command::SpawnMeat { pos } => {
                let e = self.params.plant_max_energy * 2.0;
                self.add_meat(Vec3::from(pos), e);
            }
            Command::SpawnObstacle { pos, shape, dynamic, yaw } => {
                self.add_obstacle(Vec3::from(pos), shape, dynamic, yaw);
            }
            Command::SpawnCreature { pos, from, count } => {
                for _ in 0..count.max(1) {
                    let at = pos.map(Vec3::from);
                    let source = from.and_then(|id| self.creature(id)).map(|c| (c.genome.clone(), c.generation));
                    match source {
                        Some((mut g, generation)) => {
                            let p = self.breeding_params();
                            g.mutate(&mut self.rng, &p);
                            let at = at.unwrap_or_else(|| self.random_land_point(4.0));
                            let e = p.start_energy;
                            self.spawn_creature(g, at, p.rain_height, e, generation + 1, None);
                        }
                        None => {
                            self.rain_creature(at);
                        }
                    }
                }
            }
            Command::SpawnGenome { genome, generation, pos, count } => {
                self.drop_genome(&genome, generation, pos.map(Vec3::from), count);
            }
            Command::RemoveCreature(id) => self.kill(id, DeathCause::Removed),
            Command::KillCreature(id) => self.kill(id, DeathCause::Killed),
            Command::FeedCreature(id) => {
                if let Some(c) = self.creature_mut(id) {
                    c.energy = c.max_energy;
                    c.health = c.max_health;
                }
            }
            Command::RemovePlant(id) => self.plants.retain(|f| f.id != id),
            Command::RemoveMeat(id) => self.meat.retain(|f| f.id != id),
            Command::RemoveObstacle(id) => self.remove_obstacle(id),
            Command::Select(id) => self.selected = id,
            Command::GrabCreature { creature, part, local, target } => {
                let body = self
                    .creature(creature)
                    .and_then(|c| c.parts.get(part).or(c.parts.first()))
                    .map(|p| p.body);
                self.grab = body.map(|body| Grab {
                    body,
                    local: Vec3::from(local),
                    target: Vec3::from(target),
                    creature: Some(creature),
                });
            }
            Command::GrabObstacle { obstacle, local, target } => {
                let body = self
                    .obstacles
                    .iter()
                    .find(|o| o.id == obstacle && o.dynamic)
                    .map(|o| o.body);
                self.grab = body.map(|body| Grab {
                    body,
                    local: Vec3::from(local),
                    target: Vec3::from(target),
                    creature: None,
                });
            }
            Command::DragTo(t) => {
                if let Some(g) = &mut self.grab {
                    g.target = Vec3::from(t);
                }
            }
            Command::Release => self.grab = None,
            Command::ClearCreatures => {
                for c in &mut self.creatures {
                    c.dead = Some(DeathCause::Removed);
                }
            }
            Command::ClearPlants => self.plants.clear(),
            Command::ClearObstacles => {
                let ids: Vec<u64> = self.obstacles.iter().map(|o| o.id).collect();
                for id in ids {
                    self.remove_obstacle(id);
                }
            }
        }
    }
}
