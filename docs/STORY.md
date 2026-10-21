# NeuroHunt - Story & Chapter Design

**Genre:** sci-fi stealth survival, tactical puzzles, adaptive enemy AI.
You are trapped inside an abandoned AI research facility. A robotic hunter is searching for you, and every time you play it adapts to your habits.

| Chapter | Status | Summary |
|---|---|---|
| 1 - The Awakening | **Playable** | Find the keycard, restore the terminal, escape Sector A. The hunter has limited vision and hearing: it cannot see through walls or teleport. |
| 2 - It Knows Your Tricks | Partly in (mechanics) | Repeated hiding spots and escape routes become unreliable; distractions (thrown bottles) are in. Light switches, doors and false alarms are planned. |
| 3 - The Counterattack | Planned | Cameras and security systems: study patrol patterns, disable sensors, lead the hunter away from your objective. |
| 4 - The Final Hunt | Planned | Emergency lockdown. The hunter guards access routes; you choose to escape alone, disable the AI, or expose the experiment. Choices decide the ending. |

## Chapter 1 - The Awakening (implemented)
You regain consciousness in the Cryo Lab of Sector A. Objectives:
1. Pick up the **keycard** (Reactor Access).
2. Hold **E** at the **terminal** (Control Room) to restore power. This trips the **alarm**, which the hunter can hear and which it observes as a feature.
3. Reach one of the two **exits** (Server Hall or Med Wing).

Mechanics: sneak (quiet, slow), walk, sprint (loud, uses stamina), hide in lockers, throw bottles to create false noises. The hunter remembers your favourite lockers and exits across sessions (SQLite), and TAB shows what it has learned.

## Chapter 2 - It Knows Your Tricks
Uses the same behaviour model on new layouts with more lockers and loops. Planned additions: light switches that change visibility (an observation feature already reserved), doors that block line of sight, and false alarms that the hunter must learn to discount (the "false investigations" metric already tracks this).

## Chapter 3 - The Counterattack
Security room with camera feeds showing the hunter's live vision cone; temporary sensor jamming (reduces its vision range for N ticks). The player can deliberately lure the hunter using its learned expectations against it.

## Chapter 4 - The Final Hunt
Lockdown: the hunter's reward is retrained with a "guard" objective (keep access routes covered). Three endings: escape alone, shut the AI down at its core, or upload the experiment logs.
