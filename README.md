# evolve

A 3D evolution simulator. Creatures with randomly built bodies (segmented trunks, multi-jointed
legs, arms, tentacles and paddles) live on an island in an ocean, with hills, plants, weather and a
day/night cycle. Each one is driven by a neural network. They search for food and water, breed,
fight and eat the remains, and over generations they learn to move their bodies to get around,
on land and in the water. Everything physical runs through a real physics engine.

![overview: the island at evening under clouds, with the parameter and statistics panels](docs/overview.png)

![following a creature: its inspector and a live diagram of its brain](docs/inspector.png)

![a storm: rain, haze, whitecaps and lightning](docs/storm.png)

![the director at sunset, hands-free](docs/director.png)

![the zoo: side views of every stored creature, and the selected one in 3D](docs/zoo.png)

## Run it

```
cargo run --release -p evolve
```

The first build compiles Bevy and takes a few minutes. Options:

```
cargo run --release -p evolve -- --load ~/.local/share/evolve/saves/world.evo
cargo run --release -p evolve -- --seed 42 --speed 4 --set min_population=80 --set water_level=8
```

Sound needs the ALSA development files to build (`sudo apt install libasound2-dev` on
Debian/Ubuntu). Without them, the build fails rather than running silently.

The window opens through XWayland. For a native Wayland window, install `libwayland-dev` and
build with `--features wayland`.

A fresh world starts with random bodies that mostly flail and starve. Real walking usually
appears after 40–60 simulated minutes, which is a few minutes of wall time at **Max** speed. To
skip ahead, evolve a world with the headless runner, which runs as fast as the CPU allows, prints
progress, and saves the result for the app to load:

```
cargo run --release -p evolve-sim --bin evolve-headless -- --minutes 120 --report 300 --save ~/.local/share/evolve/saves/evolved.evo
```

## Controls

| | |
|---|---|
| Right-drag / middle-drag / scroll | orbit / pan / zoom |
| WASD, Q/E | move, turn |
| F | follow the selected creature (opens its brain view) |
| C | director: the camera runs itself (any camera input takes over again) |
| M | mute; the volume slider is in the toolbar |
| 1-7 | tools: select and drag, plant food, meat, rock, boulder, drop creature, remove |
| Left click | use the tool. With Select, click a creature to inspect it, or hold and drag to carry it (scroll lifts it) |
| Space, `.` | pause, single step |
| + / - | speed (¼× up to Max) |
| F5 / F9 | quick save / quick load |
| F12 | screenshot |
| Delete, Esc | remove selected creature, deselect |

**The director** (toolbar button, C, or start with `--director`) is a camera for just watching.
- *What it picks:* it looks for something worth seeing and frames it. Candidates are a kill, a
  birth, the first of a new generation, a fight, a creature in trouble in the water, a splash, the
  best swimmer, the fastest steady walker, the most evolved creature, the biggest brain, the
  strangest body, or the island itself at sunrise, at sunset, at night and in storms.
- *How it moves:* it circles the subject slowly, then glides to the next thing, flying up and over
  for long moves. It follows the subject's center of mass on a damped spring (as does F), so a
  creature thrashing about doesn't shake the view.
- *Variety:* it avoids repeating itself, and cuts early for a kill, a birth or a record somewhere
  else.
- *Screen:* a caption says what you're looking at, and the panels hide for a clean screen (tick
  "panels" to keep them).
- *Taking over:* moving the camera yourself, or pressing C, hands control back.

**Sound** is synthesized as it plays; there are no sound files.
- *Ambience:* wind that rises with the weather and the camera's height, rain, and surf where the
  view takes in the shoreline. There are no birds or insects here; the only voices are the
  creatures' own.
- *Thunder:* follows each lightning flash after the time sound takes to travel from it, with a
  crack when close, plus distant rumbles in storms.
- *Calls:* each creature's calls in its genetic voice: FM tones that can be pure, metallic or
  bubbling, wavering or steady, clean or rasping, gliding up or down with the pitch its brain
  picks.
- *Events:* thuds when bodies hit the ground (louder for harder falls, deeper for heavier parts),
  bites, kills, birth chimes, a rising arpeggio for a new generation, splashes and drowning
  bubbles.
- *Distance:* events are full strength within 6 m of the camera and fade as 1/distance beyond:
  10 times quieter at 60 m, as sound does in open air. They come from where they happen (left or
  right).
- *Balance:* every sound has a target loudness, measured by a test with the broadcast loudness
  standard (ITU BS.1770, LUFS) and kept within 1.5 dB. Close thunder is loudest (−12), then
  hard landings and kills (−16), bites and splashes (−20), and a gale (−22). Heavy rain sits under
  everything at −27, and birds and crickets lower still.
- *Device:* plays through PipeWire or PulseAudio when available, so it follows the desktop's
  choice of speakers, and reconnects by itself if the device reports an error, or stops asking for
  sound for 4 seconds after it had started. (A device can take a while to start.)
- *Controls:* M or the toolbar mutes; the slider next to it sets the volume. `--mute` starts
  silent, and screenshot runs are always silent.

**Leaders** (in the right panel) names who's ahead right now: top score, oldest, most evolved,
biggest brain, fastest, farthest travelled, biggest eater, most children, best hunter, heaviest
and best swimmer. Click one to go and follow it.

**The brain view** opens when you follow a creature (F, or Follow in its inspector), or with the
inspector's Brain view button.
- *Layout:* the senses its body uses are on the left and the actions it can take on the right. The
  (up to) 40 hidden neurons most tied to them sit in a ring between.
- *Connections:* the strongest connections are drawn in orange (excite) or blue (inhibit), thicker
  for stronger weights and brighter while signal is flowing through them.
- *Live:* nodes light up with their current activity. Hover a node for its name and value.

**The zoo** (toolbar button, or start with `--zoo`) keeps the best creatures from every world
across sessions, in `~/.local/share/evolve/zoo`. Every `zoo_interval` simulated seconds (default
5 minutes), each world, background worlds included, stores its current best genome, and the
weakest unpinned ones are pruned past `zoo_max_entries`. **Add to zoo** in a creature's inspector
stores (and pins) one by hand. From the Zoo window you can **Drop** copies from the sky into this
world, a background world or every world, or **Place** them where you click. Like every creature
that appears in a world, each copy is a slightly mutated variant (a child of the zoo creature).
Drop 5 or 10 at once if you want them to breed: they're close enough kin to mate. Load another world
and its zoo is still there. A new world starts with `zoo_seed_fraction` (70%) of its population
from the zoo, in pairs of copies of randomly chosen creatures so they can breed, and with the
same genomes in its gene pool. The rest are random. Zoo scores worked out before the scoring
change are marked with a * and listed after newer ones. When the zoo is full (500), those go
first, then the weakest; pinned ones never.
- Pruned and removed creatures go to `zoo/trash` (the newest 500 are kept there), not deleted.
- Only one evolve at a time can change the zoo. A second one (another window, or
  `evolve-headless` with `EVOLVE_ZOO_DIR` pointing at the real zoo) can read it but stores nothing.
- If `index.json` can't be read, it is renamed `index.json.bad-<time>` and the zoo is rebuilt
  from the genome files (scores start at 0, marked *).
- Each row shows a side view of the creature's body.
- Click a row to see it in 3D, spinning slowly; drag the picture to turn it.
- Joints are marked: yellow for hinges, pink for swivels.
- Next to the picture: its body segments, limb segments, joint kinds, brain size, diet and where it
  came from.

The right panel lists the species (clusters of creatures that can interbreed) with their size,
generation, diet, limbs and brain size. Click one to follow its most advanced member.

Every parameter is in the left panel and takes effect immediately. The ones marked ⟳ shape the
terrain and apply when you press **Regenerate terrain**, which keeps all the life. **New world**
starts over. **Save / Load** lists the worlds in `~/.local/share/evolve/saves`. A save holds the
complete state, physics included, so a world resumes mid-stride exactly where it was.

## How it works

**Genome.** The body is a torso (the head end, with eyes and mouth) followed by a spine of 0 to
5 more segments, so a creature can be one block, a caterpillar or a snake. The spine gene sets the
segments' size and taper, which way they bend (side to side or up and down), their range and
strength, and whether the joints are hinges or swivels. Six limb genes follow. Each can be switched
on or off. Each grows from any body segment, or from the tip of an earlier limb. Each has 1 to 3
segments (thigh, shin, foot; upper arm, forearm, hand), with tapering size and a resting bend at
the knees. A gene also sets the limb's direction, hinge axis, range of motion, motor strength, and
whether its tip is a flat paddle. A limb can be *mirrored* to make a symmetric pair, and limbs
hanging off a mirrored limb are mirrored with it. A body is capped at 24 parts. Traits cover body
size, diet (plants or meat), built-in leanings toward eating, mating and attacking, armor, lung
capacity, the frequency of an internal rhythm, and how fast the brain learns. Lineages have their
own color hue, which drifts slowly.

**Joints.** Every joint has a genetic range of motion that the physics enforces. That takes two
constraint passes per solver iteration (`solver_passes`): with one, a thrashing snake could bend
a joint a radian past its limit.
- *Hinges* bend about one axis. They are the cheap kind. A limb's knees bend mostly one way (a
  quarter of the range the other way), like real knees and elbows.
- *Swivels* bend about two axes, with twisting along the bone locked: a hip or shoulder, or, all
  the way down a limb, a tentacle. They're more flexible, but each unit of work costs
  `swivel_energy_cost` (2x) what a hinge's does.
- *Speed:* the angle a joint aims for moves at most `max_joint_speed` (8 rad/s for a half-meter
  segment; longer segments slower, shorter ones faster, like pendulums). The brain can change its
  mind 30 times a second, but a limb swings rather than snapping back and forth in a blur.

Every joint axis has its own fixed brain channel tied to its gene, so growing a knee on one leg
never rewires another.

**Brain.** A recurrent network: every hidden neuron feeds back into every other one each tick,
so the brain keeps a state between ticks (memory, rhythms, plans). Weights are stored as f16.
Creatures made from scratch start with `brain_min_neurons` (8). Brains then grow or shrink
through mutation, by about 20% at a time (`brain_resize_rate`, with a slight upward bias set by
`brain_growth`), up to `brain_max_neurons` (2048; the slider goes to 8192). Crossover lines
parents up neuron by neuron. A brain costs `brain_energy_cost` per million connections per
second. Connections grow with the square of the neuron count: a 2048-neuron brain has 4.3 million
of them and costs about as much as resting metabolism, so a big brain has to pay for itself.
- *Senses (128):*
  - the rhythm, energy, health, oxygen, thirst and age;
  - a mating drive: an urge that builds while the creature is mature, past its cooldown and fed
    enough to breed (over `mate_drive_time`, faster with a high libido gene), fades when it goes
    hungry, and resets after mating;
  - hearing: the direction and loudness of the loudest call it can hear, its pitch, and how much
    the caller's voice is like its own;
  - how much earth it's carrying;
  - posture and speed;
  - the direction and closeness of the nearest plant, meat, open water and creature;
  - how related that creature is, its size and aggression;
  - the direction of the nearest potential mate (mature and close enough genetically);
  - three obstacle rays, pain, how submerged it is, and three memory values;
  - every joint angle (82 channels, most of them unused by any one body).
- *Actions (92):* a target angle for every joint axis (82 channels), eat, attack, mate, call
  (and its pitch), dig, pile, and memory.
- *Voices:* every creature has a voice of its own, set by four genes: natural pitch, timbre (how
  inharmonic and bright), rasp and warble. Kin sound alike.
  - *Calling:* a creature calls when its brain says so, louder or softer and pitched up or down.
    A full call carries `call_range` (40 m) and costs `call_cost` energy.
  - *Hearing:* creatures hear the call that sounds loudest from where they are.
  - *Meaning:* what calls come to mean (a mating call, a warning, nothing) is up to evolution.
  - *Old creatures:* upgraded from older versions they start silent, with a voice of their own
    worked out from their genome.
- *Lifetime learning:* creatures learn while they live, by trial and reward (reward-modulated
  node perturbation with eligibility traces):
  - *Trying things:* each joint's movement wanders a little around what the brain asks for (the
    `explore` gene, scaled by `exploration`).
  - *Remembering what was tried:* each output connection keeps a fading trace of the recent
    variations and the brain activity behind them (the `trace_time` gene: how many seconds back
    a reward can reach).
  - *Learning from surprise:* when things go better than the creature's running average, the
    traced variations are written into a learned layer on top of the inherited output weights;
    when worse, the opposite.
  - *What feels good:* eating, mating and drinking in proportion to need; getting closer to food
    when hungry or water when thirsty (weighted by the `seek` gene). Pain feels bad.
  - *How long it lasts:* lessons fade back toward the inherited wiring over `learned_memory`
    (half-life 5 minutes). The `plasticity` gene sets how fast each creature learns, and
    `learning_rate` scales everyone.
- *Inheriting lessons:* when a creature breeds, `learned_inheritance` (half, by default) of what
  it has learned is written into the genes it passes on. The same goes for a life good enough
  for the gene pool, so creatures that fall from the sky inherit too. Lineages can build on what
  their ancestors figured out, while evolution still decides how much each one explores, how
  far back it remembers, and what it finds rewarding.

**Energy and water.**
- *Energy* comes only from eating: plants, or meat in proportion to the diet gene.
  - *What it costs:* it drains with body mass (mass^0.75), brain size, armor, lung capacity,
    calling, and the mechanical work the joint motors actually do (double for swivels).
  - *Motor work:* until 2026-10-22 a bug meant motor work cost nothing, so flailing was free and
    tentacles were no dearer than hinges. Now ordinary movement costs little and a creature
    thrashing at 80 W per kg pays about twice its resting metabolism.
  - *Starving:* a creature with no energy starves.
- *Breath:* muscles burn oxygen in proportion to their power per kg (`exertion_oxygen`), and a
  creature breathes it back in while its head is above water (`breath_rate`, scaled by the lung
  gene). Hard work can outpace breathing. Below 30% breath its muscles weaken, so a sprinter must
  pace itself. Underwater it holds its breath (`breath_time`), and drowns when that runs out.
  Measured on an evolved world, creatures on land are short of breath about 6% of the time.
- *Thirst:* a full creature runs dry in `thirst_time` (6 simulated minutes) and then loses health.
  Putting its mouth in water refills it in about two seconds, and rain helps a little. Any water
  will do.
- *Health* drops from bites, from drowning (holding breath underwater too long) and from thirst,
  and regenerates slowly.

**Breeding.** A mature, willing creature with enough energy that is near a genetically close
(`species_threshold`), willing partner produces a child. The partner only needs half the energy
bar. The genomes cross over, then mutate, and
each parent gives the child part of its energy. Genetic distance splits the population into
species that can't interbreed. Breeding alone is an optional parameter.

**Combat.** A creature that touches another with its torso (its head; the eyes show which end)
and wants to attack bites it. Damage depends on the size ratio and the target's armor. Biting
costs energy and has a cooldown, and the target can bite back. The dead leave meat where their
body parts were, and the meat rots over time.

**Rain and the gene pool.** When the population drops below `min_population`, creatures fall
from the sky. Most are bred from a hall of fame of the most successful genomes seen recently,
and the rest are random.

A life is scored by what it achieved, in "plants' worth":
- *Eating:* 1 per plant's worth of food or meat.
- *Hunting:* 3 per kill.
- *Breeding:* 0.5 per child, plus 1 per plant's worth of energy given to children.
- *Travelling:* 1 per 50 m of net progress.

Lasting long earns nothing by itself. Children must start with at least `min_child_energy`.
Before both rules, tiny limbless specks that bred where they landed topped the rankings: their
energy store was so small that one bite made them "well fed" enough to breed, and every child
scored 4 points. The inspector shows a creature's score and its breakdown. The director and the
species panel spotlight the best scorer.
Distance counts net progress over 5-second windows, because an earlier per-second tally ended
up rewarding creatures that just flailed in place. This keeps
evolution moving until the population can sustain itself.

**World.**
- *Terrain:* an island in an ocean. The coast follows the hills, with headlands and bays
  (`island_size`, `coast_roughness`). The terrain is a noise heightfield that is also the physics
  collider, with water at an adjustable level. Plants grow best just above the shoreline, seed
  nearby, and die when flooded.
- *Day and weather:* the sun rises and sets over `day_length` (15 simulated minutes; nights take
  about a third of it). Clouds, rain and storms come and go on their own (`weather_period`,
  `storminess`).
  - Rain makes plants grow faster and gives a little water.
  - Wind pushes on creatures with real air drag, up to about 20 m/s in a storm.
  - The weather is a function of the world's clock and seed, so a saved world keeps its sky.

**Earthworks.** Creatures can reshape the island.
- *Digging and piling:* a creature digs earth out at its mouth when its brain's dig output is
  up, carries it (up to `soil_per_kg` of its body mass), and piles it back wherever its pile
  output says.
- *Limits:* earth is conserved and moving it costs energy (`dig_cost` per m³). Ground can be dug
  `dig_depth` (3 m) below and piled `pile_height` (3 m) above where the world made it.
- *What it can build:* pits, ponds where a hole goes below sea level, mounds, walls, ramps and
  dams. There are no tunnels or overhangs: the ground stays a smooth surface, reshaped in place.
- *What follows:* the physics ground, plants, water and rendering follow the new shape. Freshly
  moved earth shows as bare soil.
- *Senses:* the brain feels how full its load is.
- *Starting out:* random creatures start with digging off and upgraded ones with it silent, so
  evolution has to find it worth the effort.

**Swimming.** Water gives buoyancy and costs breath. Drag acts at two points along every body
part, so a limb sweeping about its joint feels more push at its fast end. Drag has a linear part
and a pressure part that grows with the square of speed (`swim_drag`). A fast stroke therefore
pushes much harder than a slow recovery, which is what lets paddles, flippers and wriggling bodies
swim. Drag is stronger across a part than along it, so a flat paddle swept face-first works like
an oar.

**Older saves and zoo creatures.**
- Saves (EVOLVE03) and zoo genomes from before multi-jointed bodies still load.
- Their brains are rewired onto the new layout so they behave exactly as before. A test checks
  this.
- Creatures in an old save are rebuilt from their genomes where they stood.

## Hardware: CPU, GPU, NPU

At **Max** speed the program is built to keep the whole machine busy.

- **GPU, brains:** every creature's brain runs on the GPU (`gpu.rs`, `brain.wgsl`). All brains of
  a world are evaluated in two batched compute dispatches per brain tick: one for every hidden
  neuron of every creature, one for every output plus the learning step. Results are read
  back one tick later, so the GPU thinks while the CPU runs physics. Brains live in growable GPU
  buffers, and `gpu_memory_gb` (default 8) caps their total memory across all worlds; a brain
  that doesn't fit is trimmed. The kernels match the CPU reference (`brain::step_cpu`) to within
  5×10⁻⁶. 50 brains of 2048 neurons take 0.52 ms per tick, about 830 GB/s, close to the RTX
  5080's memory bandwidth. Without a GPU, or with `EVOLVE_CPU_BRAINS=1`, brains run on the CPU.
- **GPU, drawing:** Bevy on wgpu/Vulkan, in HDR with bloom and filmic tonemapping, with custom
  shaders in `crates/evolve/src/shaders`:
  - *Sky:* a sky dome with sunrise and sunset colors, a sun and moon, stars, and clouds that drift
    with the wind and darken in storms.
  - *Ocean:* Gerstner waves that grow with the wind, depth-tinted from turquoise shallows to deep
    blue, with shoreline and whitecap foam and rain ripples, reaching the horizon.
  - *Land:* island terrain shading with wet sand, rock and cloud shadows.
  - *Plants:* trees and bushes that sway in the wind.
  - *Weather:* rain streaks, and lightning with flashes.
  - *Light:* the sun and moon light the scene, with shadows that scale with the camera's distance.

  The weather and time of day come from the world, smoothed so they don't flicker at Max speed.
  `--time-of-day F` and `--weather clear|cloudy|rain|storm` override what's shown, for
  screenshots. A frame costs about 1.5 to 3 ms uncapped (under 1 ms of it on the GPU), so the
  frame rate is just the monitor's refresh rate and the GPU stays free for brains.
- **CPU:** one world's physics stops getting faster at about 8 threads (rapier: 4.3 ms per step
  on 1 thread, 1.2 ms on 8, still 1.2 ms on 24). So the rest of the cores run **background worlds**
  (`parallel_worlds`, default about one per 4 cores): independent copies of the simulation on
  other landscapes, each in its own thread pool. Every `migration_interval` simulated seconds the
  best genomes migrate between worlds through a shared board. The world you watch keeps
  receiving the best bodies and brains evolved anywhere (an island-model genetic algorithm). On
  this 24-core machine, 6 background worlds ran about 7,400 steps per second in total, against
  about 1,300 for one world alone, without slowing the watched world. The Performance panel shows
  CPU use, GPU busy time and brain memory live.
- **Balancing CPU and GPU:** physics fills the CPU long before small brains fill the GPU, so brain
  size is steered to fill it.
  - *How it's steered:* at Max speed, the runner adds up every world's GPU time on brains and
    compares it with `gpu_target` (50% by the GPU's own timer, which shows as roughly 70% in
    nvidia-smi).
  - *Below the target:* brains get cheaper to run (a floating energy price) and tend to grow
    when they mutate.
  - *Above it:* they get dearer and tend to shrink.
  - *Who decides:* evolution still picks which brains do well. The steering only sets the economy
    and the drift. The Performance panel shows the GPU share and the going price.
  - *Measured* with six background worlds at Max: brains of about 45, 320 and 630 neurons kept
    the GPU about 7%, 14% and 40% busy by its own timer (nvidia-smi read 29–39%, 43% and 59%),
    with no loss of speed. Around a thousand neurons reaches the target.
  - *Why the readings differ:* system monitors read higher than the GPU's own timer, because they
    count any moment a kernel is active.
  - *Turning it off:* set `gpu_target` to 0 for a fixed brain price.
  - *Balance run:* six background worlds at Max for 14 minutes. Brains grew from about 45 to about
    600–800 neurons by minute 8, then settled around 350–650. The GPU's own timer stayed near the
    target, nvidia-smi read 84–97% (that run used a 70% target), and the CPU stayed at about 17
    cores with no loss of speed.
- **NPU:** still not used. With brains this big, the work is large batched matrix arithmetic the
  GPU does at hundreds of GB/s. The NPU wants static compiled shapes, while every creature here
  has a different brain size and new weights at every birth.

## What tuning taught

The default parameters come out of a series of headless runs, compared across three seeds rather
than single runs, because evolution's "takeoff" moment varies a lot from run to run.

**Caveat:** every lesson below was learned while a bug made motor work free (fixed 2026-10-22).
Motion now costs energy, so these conclusions may no longer hold and deserve re-testing.

- **Measure progress, not motion.** Distance was first tallied once a second, and the gene pool
  promptly selected creatures that flailed violently in place. Distance now counts net progress
  over 5-second windows, and median travel went from 0.35 m to about 4 m per 10 seconds.
- **Don't subsidize size.** Raindrops used to start with energy in proportion to body mass, so
  evolution bred 300 kg creatures that sat still and starved slowly. Starting energy is now a
  fixed amount, and evolved bodies settle near 100 kg.
- **Easier isn't faster.** Cheaper metabolism, richer food and cheaper breeding all slowed the
  evolution of walking. Longer lives mean fewer deaths, so the gene pool sees fewer evaluations,
  and cheap breeding rewards sitting next to a neighbor more than finding food. With the cheapest
  breeding, walking failed to emerge in 2 of 3 seeds within 90 minutes.
- **Recurrent brains beat feed-forward ones.** Across three seeds at 90 minutes, switching to the
  recurrent brains raised average distance per lifetime from 58 m to 77 m and food from 28 to 37,
  with births holding steady.
- **Lifetime learning has to be fed the right signals.**
  - *Plain Hebbian:* the first version strengthened whatever the brain did at the instant of a
    reward. Learned changes stayed at 2–3% of the inherited weights, and switching it off made no
    measurable difference.
  - *Twin test:* the rebuilt version, with exploration, traces and surprise, was measured by
    dropping the gene pool's 48 genomes into a world twice, once learning and once not.
  - *Eating as a reward hurt:* when eating itself was a reward, the learners ate less (41 vs 60
    at four times the rate). Meals happen standing still, so crediting the leg movements just
    before them scrambled good gaits.
  - *Approach and pain:* learning movement only from closing in on what's needed (and from pain)
    removed that harm. Learning now reaches 25–30% of the inherited weights in 20 minutes.
  - *No clear gain yet:* in 8-minute twin runs the learners are not clearly better (51.6 vs 53.0,
    55.0 vs 45.6 and 46.4 vs 56.4 across three settings, within noise). Lives are short (about
    two minutes in that world), and the real payoff, lessons passed down and built on, needs
    long evolution runs to measure.
- **Help with mating backfires.** Dropping raindrops next to compatible relatives
  (`rain_near_kin`) raised births in one seed but stopped walking from evolving in two of three.
  Breeding without moving gets rewarded, so it's off by default.
- **One thread pool for many worlds wastes the machine.** With all worlds sharing one pool, eight
  of them ran about 4,000 steps per second. With a pool per world, seven ran about 7,400.

Where the defaults stand after 3 simulated hours (it varies a lot from world to world): food
intake covers about 40–50% of what a creature spends, and births are about 20–40% of all
arrivals (the rest rain in). Mating is limited mostly by proximity: when a creature is ready, a
willing partner is in range well under 1% of the time. The population is rain-sustained until
walking and foraging evolve, and it leans more on births the longer a world runs. The background
worlds exist to speed that up.

## Layout

```
crates/evolve-sim   headless core (no Bevy): terrain, genome, brain (CPU reference), GPU brains
                    (gpu.rs, brain.wgsl), body building, physics, ecology, background worlds
                    (islands.rs), the zoo, save/load, the sim thread, the evolve-headless binary
crates/evolve       the Bevy app: sky, ocean, terrain, plants and weather (env.rs, sky.rs, water.rs,
                    trees.rs, weather_fx.rs, shaders/), entity mirroring, camera and director,
                    sound (audio.rs), zoo preview, tools, egui UI
```

Tests: `cargo test --release -p evolve-sim`. These check:
- random body plans (including snakes, knees, tentacles and paddles) and the physical stability of
  dozens of random creatures;
- that snakes, tentacles and legs stay within their joints' range;
- thirst and drinking;
- that old brains behave exactly the same after the upgrade;
- that the learning rule learns (a toy task where reward follows one output), that the GPU hands
  back exactly what a brain learned, and that parents pass on the set fraction of it;
- the neuron bounds under mutation;
- GPU and CPU brains agreeing;
- death and meat, rain, user commands and terrain regeneration;
- the zoo;
- the weather's day length and rain statistics;
- the sound mix (each sound within 1.5 dB of its target loudness), and that a fall from the sky
  lands with a thud;
- an exact save/load round trip.
`cargo test --release -p evolve-sim --test diag -- --ignored --nocapture` prints tuning
diagnostics.

Performance probes: `--no-vsync`, `--no-shadows` and `--no-ui`, combined with `--screenshot`,
print the mean frame time at exit.

The app also has a scripted self-check, which grabs and drags the creature at the screen center
and screenshots the result:

```
cargo run --release -p evolve -- --load world.evo --pick-test --screenshot out.png --frames 420
```

## License

MIT. See [LICENSE](LICENSE).
