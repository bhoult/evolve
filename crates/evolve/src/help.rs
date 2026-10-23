//! The help window: a guide to the controls and to every dynamic of the world, one topic at a
//! time. Numbers come from the current params, so the text stays true when they're changed.

use bevy_egui::egui::{self, RichText};
use evolve_sim::SimParams;
use evolve_sim::genome::{MAX_LIMB_GENES, MAX_PARTS, MAX_SPINE};

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Topic {
    #[default]
    Start,
    Controls,
    Watching,
    Bodies,
    Brains,
    Learning,
    Energy,
    Food,
    Water,
    Breeding,
    Fighting,
    Evolution,
    Land,
    Weather,
    Voices,
    Worlds,
}

impl Topic {
    const ALL: [Topic; 16] = [
        Topic::Start,
        Topic::Controls,
        Topic::Watching,
        Topic::Bodies,
        Topic::Brains,
        Topic::Learning,
        Topic::Energy,
        Topic::Food,
        Topic::Water,
        Topic::Breeding,
        Topic::Fighting,
        Topic::Evolution,
        Topic::Land,
        Topic::Weather,
        Topic::Voices,
        Topic::Worlds,
    ];

    /// The topic whose title starts with `name` (ignoring case).
    pub fn named(name: &str) -> Option<Topic> {
        let name = name.to_lowercase();
        Topic::ALL.into_iter().find(|t| t.title().to_lowercase().starts_with(&name))
    }

    fn title(self) -> &'static str {
        match self {
            Topic::Start => "Getting started",
            Topic::Controls => "Controls",
            Topic::Watching => "Watching",
            Topic::Bodies => "Bodies and muscles",
            Topic::Brains => "Brains and senses",
            Topic::Learning => "Learning in life",
            Topic::Energy => "Energy",
            Topic::Food => "Plants and meat",
            Topic::Water => "Water and breath",
            Topic::Breeding => "Breeding",
            Topic::Fighting => "Fighting",
            Topic::Evolution => "Evolution and scores",
            Topic::Land => "Land and water",
            Topic::Weather => "Day and weather",
            Topic::Voices => "Voices",
            Topic::Worlds => "Worlds and the zoo",
        }
    }
}

pub fn help_window(ctx: &egui::Context, open: &mut bool, topic: &mut Topic, p: &SimParams) {
    egui::Window::new("Help")
        .open(open)
        .default_size([760.0, 560.0])
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(150.0);
                    for t in Topic::ALL {
                        ui.selectable_value(topic, t, t.title());
                    }
                });
                ui.separator();
                ui.vertical(|ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("help_text")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.heading(topic.title());
                            ui.add_space(4.0);
                            page(ui, *topic, p);
                        });
                });
            });
        });
}

fn para(ui: &mut egui::Ui, text: impl Into<String>) {
    ui.add(egui::Label::new(text.into()).wrap());
    ui.add_space(4.0);
}

fn bullet(ui: &mut egui::Ui, text: impl Into<String>) {
    ui.horizontal_top(|ui| {
        ui.label("•");
        ui.add(egui::Label::new(text.into()).wrap());
    });
}

fn sub(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(title).strong());
}

/// A number without trailing zeros.
pub fn num(v: f32) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.into() }
}

fn pct(v: f32) -> String {
    format!("{}%", num(v * 100.0))
}

/// Simulated seconds, in seconds, minutes or hours.
pub fn dur(s: f32) -> String {
    if s < 120.0 {
        format!("{} s", num(s.round()))
    } else if s < 7200.0 {
        format!("{} min", num((s / 6.0).round() / 10.0))
    } else {
        format!("{} h", num((s / 360.0).round() / 10.0))
    }
}

/// How long a piece of meat holding `e` energy lasts (it loses `decay` of itself per second plus
/// 0.02, and is gone below 0.2).
fn meat_lasts(e: f32, decay: f32) -> f32 {
    const FLAT: f32 = 0.02;
    if decay <= 1e-6 {
        return (e - 0.2).max(0.0) / FLAT;
    }
    let c = FLAT / decay;
    ((e + c) / (0.2 + c)).ln().max(0.0) / decay
}

fn page(ui: &mut egui::Ui, topic: Topic, p: &SimParams) {
    match topic {
        Topic::Start => start(ui, p),
        Topic::Controls => controls(ui),
        Topic::Watching => watching(ui),
        Topic::Bodies => bodies(ui, p),
        Topic::Brains => brains(ui, p),
        Topic::Learning => learning(ui, p),
        Topic::Energy => energy(ui, p),
        Topic::Food => food(ui, p),
        Topic::Water => water(ui, p),
        Topic::Breeding => breeding(ui, p),
        Topic::Fighting => fighting(ui, p),
        Topic::Evolution => evolution(ui, p),
        Topic::Land => land(ui, p),
        Topic::Weather => weather(ui, p),
        Topic::Voices => voices(ui, p),
        Topic::Worlds => worlds(ui, p),
    }
}

fn start(ui: &mut egui::Ui, p: &SimParams) {
    para(
        ui,
        "An island of creatures that evolve. Each is a body of blocks and limbs driven by a neural network, and nothing it does is scripted: walking, swimming, eating, breeding and fighting all have to be found, by evolution over generations and by each creature's own learning while it lives.",
    );
    para(
        ui,
        format!(
            "A life: a creature falls from the sky or is born, looks for food and water, and if it eats well, finds a partner of its own kind and breeds. It dies of hunger, thirst, drowning, bites or old age (after {}). The best lives go into a gene pool that new creatures are bred from, so later generations tend to do better.",
            dur(p.lifespan)
        ),
    );
    sub(ui, "Where to look");
    bullet(ui, "Click a creature (Select tool, key 1) to open its inspector: how it's doing, its traits and its score. Genome there explains every one of its genes; Brain view shows its network working live.");
    bullet(ui, "The right panel lists the species and the leaders (top score, fastest, best hunter...). Click one to follow it.");
    bullet(ui, "The left panel holds every parameter, and changes take effect at once. The numbers in this help are the current settings, so they follow your changes.");
    bullet(ui, "C hands the camera to the director, which goes looking for interesting moments by itself.");
}

fn controls(ui: &mut egui::Ui) {
    sub(ui, "Camera");
    bullet(ui, "Right-drag orbits, middle-drag pans, scroll zooms.");
    bullet(ui, "WASD moves, Q and E turn.");
    bullet(ui, "F follows the selected creature (and opens its brain view). C turns on the director; any camera input takes over again.");
    sub(ui, "Tools (keys 1 to 9)");
    bullet(ui, "1 Select: click a creature to inspect it, or hold and drag to carry it (scroll lifts it).");
    bullet(ui, "2 Plant food, 3 drop meat, 4 place a rock, 5 place a boulder, 6 drop a creature, 7 remove things, 8 call down a meteor, 9 set down a whirlwind.");
    sub(ui, "Keys");
    bullet(ui, "Space pauses, . steps once, + and - change speed (a quarter of real time up to Max).");
    bullet(ui, "F5 quick save, F9 quick load. Save / Load in the toolbar lists every saved world.");
    bullet(ui, "F12 takes a screenshot; R or F10 starts or stops recording a video, in the director's view too.");
    bullet(ui, "M mutes (the volume slider is in the toolbar). H opens this help.");
    bullet(ui, "Delete removes the selected creature, Esc deselects.");
    sub(ui, "Panels");
    bullet(ui, "Left: every parameter. Those marked ⟳ shape the terrain and apply on Regenerate terrain, which keeps all the life; New world starts over.");
    bullet(ui, "Right: population, charts, species and leaders. Bottom: the event log.");
    bullet(ui, "The clock at the top, in every view, shows the day and the time of day.");
    sub(ui, "Powers");
    bullet(ui, "Storm (toolbar, second row) calls a storm down on the whole island for four minutes; click it again to calm it.");
    bullet(ui, "Meteor (tool 8): click a spot and a meteor comes down on it. It digs a crater, kills what it lands on, hurts and throws everything within 16 m, scorches the plants, and burns.");
    bullet(ui, "Whirlwind (tool 9): click a spot and a whirlwind touches down there and wanders the island for 25 seconds, lifting creatures and boulders, whirling them round and flinging them out, and tearing up plants.");
    bullet(ui, "Raise land and Lower land: hold and sweep to reshape the island (the brush sets how wide). Raised out of the sea, it's new land; sunk below it, the sea comes in.");
    bullet(ui, "Spring: click to make an eternal spring. Its stream runs downhill whatever the weather and fills the hollows it reaches; click it again to stop it.");
    bullet(ui, "Earthquake: click to split the land across your view: a crack opens, one side heaves up, and everything within 70 m shakes for six seconds.");
    bullet(ui, "Land from the sky: click and a chunk of land falls there, raising a hill (out of the sea, new land) and crushing what's under it.");
}

fn watching(ui: &mut egui::Ui) {
    sub(ui, "The director (C)");
    para(ui, "A camera that runs itself. It looks for something worth seeing (a kill, a birth, a new generation, a fight, a creature in trouble in the water, the best swimmer, the fastest walker, the strangest body, sunrise, storms) and circles it, gliding from one to the next. Each shot lasts its own time, mostly 10 to 25 seconds and now and then much longer, and a caption says what you're looking at. Moving the camera yourself, or C, takes back control. Space pauses the world in any view; a notice at the top says when it is.");
    sub(ui, "The inspector");
    para(ui, "Click a creature to see its energy, health, breath, water and mating drive, its body and traits, what it has done in life and its score. Buttons follow it, feed it, clone it, kill it, remove it, or keep it in the zoo.");
    bullet(ui, "Genome: every gene, what it does, and where it sits in its possible range.");
    bullet(ui, "Feed, Kill and Remove are acts of god, and they look it: feeding is a blessing (a shaft of golden light, a glow, a choir), killing a smite (lightning out of a clear sky, then fire), removing takes the creature up in a shaft of white light.");
    bullet(ui, "Brain view: the network live. The Network tab draws the strongest connections, orange exciting and blue inhibiting; the Neural tab draws the same brain as nerve cells with spikes running down axons to the muscles.");
    sub(ui, "Leaders and species");
    para(ui, "The right panel names who's ahead now (top score, oldest, most evolved, biggest brain, fastest, farthest travelled, biggest eater, most children, best hunter, heaviest, best swimmer) and lists the species. Click any of them to follow it.");
    sub(ui, "Recording");
    para(ui, "R, F10 or the Record button saves the window, panels and overlays included, to a video in ~/Videos at 30 frames a second (no sound). The director's bar has a Record button too, showing how long it has been recording.");
    sub(ui, "Sound");
    para(ui, "Everything you hear is synthesized as it happens: wind, rain, surf and thunder, the creatures' calls in their own voices, thuds of bodies landing, bites, births and splashes. Sounds come from where they happen and fade with distance.");
}

fn bodies(ui: &mut egui::Ui, p: &SimParams) {
    para(
        ui,
        format!(
            "A body is built from its genome: a torso (the head end, with the eyes and the mouth at its front), a spine of 0 to {MAX_SPINE} more segments trailing behind it, and {MAX_LIMB_GENES} limb genes that can each be on or off. A body has at most {MAX_PARTS} parts. A creature's Genome window shows how its body comes from its genes."
        ),
    );
    sub(ui, "Shapes");
    bullet(ui, "No spine and no limbs: a single block that can only roll and twitch. A spine makes a snake (bending side to side) or a caterpillar (bending up and down).");
    bullet(ui, "A limb has 1 to 3 segments (thigh, shin, foot) and grows from any body segment or from the tip of an earlier limb. It can be mirrored into a pair, and its tip can be a flat paddle.");
    bullet(ui, "The size gene scales every part; mass grows with its cube.");
    sub(ui, "Joints");
    bullet(ui, "Hinges bend about one axis. Knees bend mostly one way, a quarter as far the other, like real knees and elbows.");
    bullet(
        ui,
        format!(
            "Swivels bend about two axes: a hip or shoulder, or, all the way down a limb, a tentacle. More flexible, but each unit of their work costs {}x a hinge's (swivel_energy_cost).",
            num(p.swivel_energy_cost)
        ),
    );
    bullet(ui, "Every joint's range of motion is in its genes, and the physics enforces it.");
    sub(ui, "Muscles");
    bullet(
        ui,
        format!(
            "Each joint has a motor that works like a servo: it holds the angle the brain asks for against any load, up to {} N·m per kg of body (motor_torque_per_kg) times the limb's strength gene, and pushes with all of it once the joint is forced {} rad off (motor_reach). So a leg on the ground can lift and push the whole body.",
            num(p.motor_torque_per_kg),
            num(p.motor_reach)
        ),
    );
    bullet(
        ui,
        format!(
            "The angle a joint aims for moves at most {} rad/s for a half-meter segment (max_joint_speed), slower for longer ones, so limbs swing rather than snap.",
            num(p.max_joint_speed)
        ),
    );
    bullet(ui, "Muscles burn energy for the work they do and oxygen for their power (see Energy and Water and breath). Short of breath, they weaken.");
    sub(ui, "Physics");
    para(
        ui,
        format!(
            "The whole body is one articulated system, so a limb pushing on the ground has to lift the body's real weight. Bodies are {} kg/m³ against water's {} (body_density, water_density), so they barely float. Ground friction is {}, gravity {} m/s². As a guard against glitches, parts are held under {} m/s and {} rad/s.",
            num(p.body_density),
            num(p.water_density),
            num(p.ground_friction),
            num(p.gravity),
            num(p.max_body_speed),
            num(p.max_body_spin)
        ),
    );
}

fn brains(ui: &mut egui::Ui, p: &SimParams) {
    para(
        ui,
        format!(
            "Each creature has a recurrent neural network that runs {} times a second (brain_hz). Every hidden neuron feeds back into every other one, so a brain keeps a state from moment to moment: memory, rhythms, plans.",
            num(p.brain_hz)
        ),
    );
    sub(ui, "Size and cost");
    bullet(
        ui,
        format!(
            "Brains made from scratch have about {} hidden neurons. They grow or shrink by about 20% in {} of births, up to {} neurons (brain_min_neurons, brain_resize_rate, brain_max_neurons).",
            p.brain_min_neurons,
            pct(p.brain_resize_rate),
            p.brain_max_neurons
        ),
    );
    bullet(
        ui,
        format!(
            "A brain costs {} energy a second per million connections (brain_energy_cost), and connections grow with the square of the neurons, so a big brain has to pay for itself. The price follows how busy the GPU is: with room to spare brains get cheaper and tend to grow.",
            num(p.brain_energy_cost)
        ),
    );
    sub(ui, "Senses (128)");
    bullet(ui, "Its own state: energy, health, breath, thirst, age, an internal rhythm, its posture and speed, every joint angle, how much earth it carries, pain, how deep in water it is.");
    bullet(
        ui,
        format!(
            "The direction and closeness of the nearest plant, meat, open water and creature within {} m (sense_radius), and of the nearest possible mate.",
            num(p.sense_radius)
        ),
    );
    bullet(ui, "About that creature: how related it is, its size and how aggressive it is.");
    bullet(
        ui,
        format!("Three obstacle rays {} m long (ray_length), ahead and to each side.", num(p.ray_length)),
    );
    bullet(ui, "The mating drive, the loudest call it hears (direction, loudness, pitch, and whether it sounds like kin), and three memory values it writes itself.");
    sub(ui, "Actions (92)");
    bullet(ui, "A target angle for every joint axis.");
    bullet(ui, "Eat, attack, mate, call (and its pitch), dig, pile earth, and the three memory values.");
    bullet(ui, "The aggression, libido and appetite genes add a built-in leaning to attack, mate and eat.");
}

fn learning(ui: &mut egui::Ui, p: &SimParams) {
    para(ui, "Besides evolving over generations, each creature learns while it lives, by trial and reward.");
    bullet(
        ui,
        format!(
            "Trying things: each joint's movement wanders a little around what the brain asks for (the explore gene, times exploration = {}).",
            num(p.exploration)
        ),
    );
    bullet(ui, "Remembering what was tried: each connection keeps a fading trace of recent variations (the trace_time gene: how many seconds back a reward can reach).");
    bullet(ui, "Learning from surprise: when things go better than usual, the traced variations are written into a learned layer over the inherited wiring; when worse, the opposite.");
    bullet(ui, "What feels good: eating, drinking and mating, in proportion to need; closing in on food when hungry, on water when thirsty, on a mate when the drive is up (the seek gene sets how much). Pain feels bad.");
    bullet(
        ui,
        format!(
            "Lessons fade back toward the inherited wiring with a half-life of {} (learned_memory). The plasticity gene sets how fast each creature learns, times learning_rate = {}.",
            dur(p.learned_memory),
            num(p.learning_rate)
        ),
    );
    bullet(
        ui,
        format!(
            "Inheriting lessons: {} of what a parent has learned is written into the genes it passes on (learned_inheritance), so lineages build on what their ancestors figured out.",
            pct(p.learned_inheritance)
        ),
    );
}

fn energy(ui: &mut egui::Ui, p: &SimParams) {
    let rest60 = p.basal_metabolism * 60f32.powf(0.75);
    para(
        ui,
        format!(
            "Energy comes only from eating. A creature stores up to {} per kg of body (energy_per_mass), and one with none left starves. Creatures that fall from the sky start with {} (start_energy), or less if that's more than they can store.",
            num(p.energy_per_mass),
            num(p.start_energy)
        ),
    );
    sub(ui, "What it costs");
    bullet(
        ui,
        format!(
            "Living: {} x mass^0.75 a second (basal_metabolism), {} a second for a 60 kg body. Armor adds up to half again, big lungs up to 30%.",
            num(p.basal_metabolism),
            num(rest60)
        ),
    );
    bullet(ui, "Thinking: the brain's cost (see Brains and senses).");
    bullet(
        ui,
        format!(
            "Moving: {} per joule of work the muscles do (motor_energy_cost), {}x that for swivels. Ordinary movement costs little; thrashing about costs about twice resting.",
            num(p.motor_energy_cost),
            num(p.swivel_energy_cost)
        ),
    );
    bullet(
        ui,
        format!(
            "Healing: {} health a second, at half an energy per point of health (health_regen).",
            num(p.health_regen)
        ),
    );
    bullet(
        ui,
        format!(
            "Calling (up to {} a second at full voice), biting ({} a bite), and moving earth ({} per m³).",
            num(p.call_cost),
            num(p.attack_cost),
            num(p.dig_cost)
        ),
    );
    bullet(ui, "Children: a child's body and its first energy come from its parents (see Breeding).");
    sub(ui, "Breath");
    para(
        ui,
        format!(
            "Muscles also burn oxygen in proportion to their power per kg ({} per W/kg each second, exertion_oxygen). Hard work can outpace breathing, and below 30% breath the muscles weaken, so a sprinter has to pace itself.",
            num(p.exertion_oxygen)
        ),
    );
}

fn food(ui: &mut egui::Ui, p: &SimParams) {
    sub(ui, "Plants");
    para(ui, "The trees and bushes are the plants, the base of all food.");
    bullet(
        ui,
        format!(
            "Each holds up to {} energy and gains up to {} a second (plant_max_energy, plant_growth), {} faster in a downpour (rain_plant_growth).",
            num(p.plant_max_energy),
            num(p.plant_growth),
            pct(p.rain_plant_growth)
        ),
    );
    bullet(ui, "They grow best on gentle, fertile slopes just above the shore, and poorly toward the peaks, on steep ground and on trampled paths. Nothing grows in a stream or a pond, and flooded plants wither.");
    bullet(
        ui,
        format!(
            "Grown plants drop seeds up to {} m away (plant_spread_radius), and {} new plants a second sprout at random spots (plant_spawn_rate), up to {} in all (plant_max_count).",
            num(p.plant_spread_radius),
            num(p.plant_spawn_rate),
            p.plant_max_count
        ),
    );
    sub(ui, "Eating");
    bullet(
        ui,
        format!(
            "With its eat output on, a creature bites the nearest plant or meat within {} m of its mouth (eat_reach), taking {} energy a second times its size gene (bite_rate). A full creature can't eat.",
            num(p.eat_reach),
            num(p.bite_rate)
        ),
    );
    bullet(ui, "The diet gene decides what it gets out of it: a plant gives (1 - diet) of its energy and meat gives diet, at least 5% each; the rest is wasted. A pure plant-eater gets nothing much from meat, and the other way round. With both in reach it eats whichever suits it better.");
    sub(ui, "Meat");
    bullet(
        ui,
        format!(
            "A dead creature leaves a piece of meat at each body part: {} energy per kg of body (meat_per_mass) plus half the energy it still held.",
            num(p.meat_per_mass)
        ),
    );
    bullet(
        ui,
        format!(
            "Meat rots: it loses {} of its energy a second (meat_decay) and a little more, and is gone when almost nothing is left. A 10-energy piece lasts about {}, a 30-energy piece {}, a 100-energy piece {}.",
            pct(p.meat_decay),
            dur(meat_lasts(10.0, p.meat_decay)),
            dur(meat_lasts(30.0, p.meat_decay)),
            dur(meat_lasts(100.0, p.meat_decay))
        ),
    );
    bullet(ui, "A body costs its parents the meat it will leave, so meat-eaters can't live on their own dead: in the end all food comes from plants, or from creatures falling from the sky.");
}

fn water(ui: &mut egui::Ui, p: &SimParams) {
    sub(ui, "Thirst");
    bullet(
        ui,
        format!(
            "A full creature runs dry in {} (thirst_time), then loses {} health a second (dehydration_damage).",
            dur(p.thirst_time),
            num(p.dehydration_damage)
        ),
    );
    bullet(
        ui,
        format!(
            "With its mouth in any water (sea, lake, pond or stream) it drinks {} of a full drink a second (drink_rate). Rain gives {} a second in a downpour (rain_drinking).",
            pct(p.drink_rate),
            pct(p.rain_drinking)
        ),
    );
    sub(ui, "Breath");
    bullet(
        ui,
        format!(
            "Head out of water, it breathes back {} of a full breath a second (breath_rate), 0.6 to 1.4 times that by its lung gene, less whatever its muscles are burning.",
            pct(p.breath_rate)
        ),
    );
    bullet(
        ui,
        format!(
            "Head under, it holds its breath for {} (breath_time) times 0.3 to 2 by its lung gene, and less if it works hard. Then it drowns, losing {} health a second (drowning_damage).",
            dur(p.breath_time),
            num(p.drowning_damage)
        ),
    );
    bullet(ui, "Big lungs cost up to 30% more resting metabolism.");
    sub(ui, "Swimming");
    para(
        ui,
        format!(
            "Water holds bodies up and drags on them at two points along every part, so a limb sweeping about its joint feels more push at its fast end. The drag has a part that grows with the square of speed (swim_drag = {}), so a fast stroke pushes much harder than a slow recovery: that's what lets paddles, flippers and wriggling bodies swim. Drag is stronger across a part than along it, so a flat paddle swept face-first works like an oar.",
            num(p.swim_drag)
        ),
    );
}

fn breeding(ui: &mut egui::Ui, p: &SimParams) {
    sub(ui, "When two creatures breed");
    bullet(
        ui,
        format!(
            "Both are mature (older than {}, maturity_age), past their cooldown ({} after the last time, mate_cooldown), and have their mate output on.",
            dur(p.maturity_age),
            dur(p.mate_cooldown)
        ),
    );
    bullet(
        ui,
        format!(
            "The one starting it has at least {} of a full energy bar (mate_energy_fraction); the partner half that.",
            pct(p.mate_energy_fraction)
        ),
    );
    bullet(
        ui,
        format!(
            "Their bodies are within {} m (mate_range), and their genes are close enough: a genetic distance under {} (species_threshold). This is what splits the population into species that can't interbreed.",
            num(p.mate_range),
            num(p.species_threshold)
        ),
    );
    bullet(
        ui,
        format!("No births happen above {} creatures (max_population).", p.max_population),
    );
    sub(ui, "What a child costs");
    let body = p.meat_per_mass * 30.0;
    bullet(
        ui,
        format!(
            "First its body, at the meat it will leave when it dies: {} energy per kg (meat_per_mass), split between the parents. A 30 kg child's body costs {} energy, {} from each.",
            num(p.meat_per_mass),
            num(body),
            num(body / 2.0)
        ),
    );
    bullet(
        ui,
        format!(
            "Then each parent gives it {} of what it has left as its starting energy (child_energy_share). If together they can't give at least {} (min_child_energy), there's no child.",
            pct(p.child_energy_share),
            num(p.min_child_energy)
        ),
    );
    bullet(ui, "So breeding needs well-fed parents. Without the body's cost, crowds of meat-eaters once lived on their own dead: every birth was a free body and every death left meat, so they gathered in heaps and bred without ever leaving.");
    sub(ui, "The child");
    bullet(
        ui,
        format!(
            "Its genes are a mix of its parents' (crossover), then mutated, and it inherits {} of what each parent learned. It's born next to its parent.",
            pct(p.learned_inheritance)
        ),
    );
    sub(ui, "The mating drive");
    para(
        ui,
        format!(
            "An urge the brain can sense. It builds while the creature could breed, over about {} (mate_drive_time; half that with the highest libido gene, twice with the lowest), fades when it goes hungry, and resets after mating.",
            dur(p.mate_drive_time)
        ),
    );
    if p.asexual {
        para(
            ui,
            format!(
                "Breeding alone is on: a creature with {} of a full energy bar can bud a mutated copy of itself, paying for all of it.",
                pct(p.asexual_energy_fraction)
            ),
        );
    } else {
        para(ui, "Breeding alone (asexual) is off.");
    }
}

fn fighting(ui: &mut egui::Ui, p: &SimParams) {
    para(ui, "Creatures fight by biting. A bite lands when a creature's torso (its head; the eyes show which end) touches any part of another while its attack output is on. There is no aiming or limb strike: ramming another head-first with attack on is a bite, and limbs and tails only help get into position.");
    sub(ui, "Damage");
    bullet(
        ui,
        format!(
            "A bite takes {} health (attack_damage), from half that for a much smaller biter up to one and a half times for one four or more times heavier.",
            num(p.attack_damage)
        ),
    );
    bullet(ui, "The victim's armor gene cuts the damage by up to 60%. Armor costs up to 50% more resting metabolism.");
    bullet(
        ui,
        format!(
            "Each bite costs the biter {} energy (attack_cost), and it can bite at most once every {} s (attack_cooldown).",
            num(p.attack_cost),
            num(p.attack_cooldown)
        ),
    );
    bullet(ui, "Health is about 100 for a 60 kg creature, less for smaller ones, and heals slowly at a cost in energy.");
    sub(ui, "The victim");
    bullet(ui, "Feels pain (a sense its brain can learn to flee from) and a learning penalty in proportion to the damage. It can bite back on the same terms.");
    sub(ui, "A kill");
    bullet(ui, "A creature whose health runs out within 5 s of a bite counts as killed by its last attacker, which scores 3 points for it (as much as eating three plants).");
    bullet(ui, "The body leaves meat where it lay, for whoever gets to it first.");
    sub(ui, "Sizing each other up");
    para(ui, "Creatures sense the nearest other creature's direction, closeness, size relative to their own, aggression and kinship, so a brain can learn to pick on smaller ones and avoid bigger ones.");
}

fn evolution(ui: &mut egui::Ui, p: &SimParams) {
    sub(ui, "Rain from the sky");
    para(
        ui,
        format!(
            "Whenever fewer than {} creatures are alive (min_population), a new one falls from {} m every {} s (rain_height, rain_interval). {} of them are bred from the gene pool, the rest are random (rain_from_pool).",
            p.min_population,
            num(p.rain_height),
            num(p.rain_interval),
            pct(p.rain_from_pool)
        ),
    );
    sub(ui, "The gene pool and scores");
    para(ui, "The gene pool is a hall of fame of the best lives lately. A life is scored by what it achieved, in plants' worth:");
    bullet(
        ui,
        format!(
            "Eating: 1 per plant's worth ({} energy) of food or meat.",
            num(p.plant_max_energy)
        ),
    );
    bullet(ui, "Hunting: 3 per kill.");
    bullet(ui, "Breeding: 0.5 per child, plus 1 per plant's worth of energy given to children.");
    bullet(ui, "Travelling: 1 per 50 m of real progress, counted over 5-second stretches so flailing in place earns nothing.");
    para(ui, "Lasting long earns nothing by itself. The inspector shows each creature's score and its breakdown.");
    sub(ui, "Mutation");
    bullet(
        ui,
        format!(
            "Each gene has a {} chance of changing in a child (mutation_rate), by about {} of its range (mutation_strength).",
            pct(p.mutation_rate),
            pct(p.mutation_strength)
        ),
    );
    bullet(
        ui,
        format!(
            "Rarer structural changes ({}, structural_mutation) add or remove limbs and neurons.",
            pct(p.structural_mutation)
        ),
    );
    sub(ui, "Generations and species");
    para(ui, "A child is one generation past the more evolved of its parents. Creatures whose genes are close enough to interbreed form a species; the right panel lists them.");
}

fn land(ui: &mut egui::Ui, p: &SimParams) {
    sub(ui, "The island");
    para(
        ui,
        format!(
            "A {} m square world with an island in the sea at {} m (world_size, water_level), with hills up to {} m high (terrain_height). The coast follows the hills, with headlands and bays. Parameters marked ⟳ reshape it on Regenerate terrain.",
            num(p.world_size),
            num(p.water_level),
            num(p.terrain_height)
        ),
    );
    sub(ui, "Rain, streams and lakes");
    if !p.hydrology {
        para(ui, "Running water is switched off (hydrology).");
    }
    bullet(ui, "Rain runs downhill. Where enough water gathers from uphill a stream runs (runoff sets how big); streams swell in rain and dwindle over the next half hour or so.");
    bullet(
        ui,
        format!(
            "Rain collects in hollows, holes dug by creatures included: {} mm an hour at full strength from everything draining into them (rain_rate). The water lies level, a full hollow spills over its lowest edge, and lakes lose {} mm an hour to evaporation (evaporation).",
            num(p.rain_rate),
            num(p.evaporation)
        ),
    );
    bullet(ui, "Lakes, ponds and streams are water like the sea: creatures drink, swim and can drown in them.");
    sub(ui, "Wear");
    bullet(
        ui,
        format!(
            "Streams slowly cut their beds deeper, faster with more water, on steeper and on trampled ground (erosion_rate), down to {} m below the original land (erosion_max_depth).",
            num(p.erosion_max_depth)
        ),
    );
    bullet(
        ui,
        format!(
            "Feet wear the ground they cross, more for heavy and fast creatures (trampling): busy ground turns bare, then to mud when wet, and plants grow poorly on it. Left alone it grows back halfway in {} (wear_recovery).",
            dur(p.wear_recovery * 60.0)
        ),
    );
    sub(ui, "Earthworks");
    bullet(
        ui,
        format!(
            "With its dig output up, a creature digs earth at its mouth and carries it (up to {} m³ per kg of body, soil_per_kg); its pile output puts it back down. Moving earth costs {} energy per m³ (dig_cost).",
            num(p.soil_per_kg),
            num(p.dig_cost)
        ),
    );
    bullet(
        ui,
        format!(
            "Earth is conserved. Ground can be dug {} m below and piled {} m above where the world made it (dig_depth, pile_height): pits, ponds below sea level, mounds, walls, ramps and dams, but no tunnels.",
            num(p.dig_depth),
            num(p.pile_height)
        ),
    );
    bullet(ui, "Creatures start with digging off, so evolution has to find it worth the effort.");
}

fn weather(ui: &mut egui::Ui, p: &SimParams) {
    if p.day_night {
        para(
            ui,
            format!(
                "The sun rises and sets every {} (day_length); nights take about a third of it.",
                dur(p.day_length)
            ),
        );
    } else {
        para(ui, "Day and night are off: it's always late morning.");
    }
    if p.weather {
        bullet(
            ui,
            format!(
                "Clouds, rain and storms come and go about every {} (weather_period). Storminess {} (0 mostly fair, 1 mostly wet).",
                dur(p.weather_period),
                num(p.storminess)
            ),
        );
        bullet(ui, "Rain makes plants grow faster, gives a little water to drink, and fills streams and lakes.");
        bullet(
            ui,
            format!(
                "Wind pushes on creatures with real air drag (wind_force = {}), up to about 20 m/s in a storm.",
                num(p.wind_force)
            ),
        );
        bullet(ui, "Storms bring lightning; the thunder follows after the time sound takes to travel.");
        bullet(ui, "The weather follows the world's clock and seed, so a saved world keeps its sky.");
    } else {
        para(ui, "Weather is off: fair skies forever.");
    }
}

fn voices(ui: &mut egui::Ui, p: &SimParams) {
    para(ui, "Every creature has a voice of its own, set by four genes: pitch, timbre (pure to metallic), rasp and warble. Kin sound alike.");
    bullet(
        ui,
        format!(
            "Calling: when its brain says so, louder or softer and pitched up or down. A full call carries {} m (call_range) and costs {} energy a second (call_cost).",
            num(p.call_range),
            num(p.call_cost)
        ),
    );
    bullet(ui, "Hearing: each creature hears the loudest call where it is: its direction, loudness, pitch, and how much the caller sounds like its own kind.");
    bullet(ui, "Meaning: what calls come to mean (a mating call, a warning, nothing at all) is up to evolution.");
}

fn worlds(ui: &mut egui::Ui, p: &SimParams) {
    sub(ui, "Background worlds");
    para(
        ui,
        format!(
            "{} more worlds evolve unseen on spare CPU cores (parallel_worlds), each with {} of this world's population (background_population). Every {} (migration_interval) their best genomes move between worlds' gene pools, so evolution runs several times faster.",
            p.parallel_worlds,
            pct(p.background_population),
            dur(p.migration_interval)
        ),
    );
    sub(ui, "The zoo");
    bullet(
        ui,
        format!(
            "Keeps the best creatures across sessions. Every {} (zoo_interval) each world stores its best genome; past {} entries (zoo_max_entries) the weakest unpinned ones go to the zoo's trash folder.",
            dur(p.zoo_interval),
            p.zoo_max_entries
        ),
    );
    bullet(
        ui,
        format!(
            "A new world starts with {} of its creatures from the zoo (zoo_seed_fraction), in pairs so they can breed.",
            pct(p.zoo_seed_fraction)
        ),
    );
    bullet(ui, "From the Zoo window you can drop copies into this or any world, or place them where you click. Add to zoo in an inspector keeps one by hand (pinned).");
    bullet(ui, "Scores earned under older rules are marked * and go first when the zoo is full.");
    sub(ui, "Saves");
    para(ui, "A save holds the whole world, physics included, so it resumes mid-stride. F5 and F9 quick save and load; Save / Load lists every saved world.");
    sub(ui, "Physics on many cores");
    para(
        ui,
        format!(
            "This world's physics is split into {} worlds stepped in parallel (physics_shards). Creatures that might touch are kept in the same one, so they still collide, fight and mate.",
            p.physics_shards
        ),
    );
}
