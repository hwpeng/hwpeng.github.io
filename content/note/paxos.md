---
title: "Paxos: Distributed Consensus"
date: 2025-11-30
area: "Systems"
summary: "A working explanation of consensus, quorum intersection, and the safety rules behind Paxos."
---

## 1. The Problem: State Machine Replication
The goal is to build a reliable system out of unreliable components. We want a cluster of servers to behave like a single, fault-tolerant machine.

* **The Mechanism:** A **Distributed Log**. If all servers execute the same commands in the same order (Sequential Consistency), they will reach the same state.
* **The Challenge:** We are in an **Asynchronous System** (no global clock, arbitrary network delays).
* **The Constraint (FLP Impossibility):** In this environment, you cannot distinguish a **crashed** node from a **slow** node. Therefore, you cannot guarantee consensus will finish in bounded time.
    * **Paxos' Choice:** It prioritizes **Safety** (Consistency) over **Liveness** (Availability). It prefers to hang/stall rather than corrupt data.

## 2. The Solution: The Paxos Algorithm
We focused on the **Synod Algorithm**, which solves consensus for a **single value** (e.g., `Log[Index 0]`). To build a full system (Multi-Paxos), we simply run this algorithm repeatedly for each slot in the log.

### Core Concepts
* **Proposer (Leader):** The active node trying to write a value. Ideally, there is only one "Distinguished Proposer" at a time to prevent livelock.
* **Acceptors:** The memory nodes. They vote on values.
* **Quorum:** A majority ($N/2 + 1$). Any two quorums must overlap by at least one node, ensuring memory persists across failures.
* **Proposal Number ($n$):** A logical clock (monotonically increasing ID). Higher numbers always preempt lower numbers, allowing a new Leader to "fire" an old, zombie Leader.

### The 2-Phase Protocol
1.  **Phase 1: Prepare (Read)**
    * **Goal:** "Lock" the Acceptors against older leaders and **discover** any partially committed data.
    * **Action:** Proposer sends `Prepare(n)`.
    * **Promise:** Acceptors promise to ignore any future request with $ID < n$. They also report the highest-numbered value they previously accepted.

2.  **Phase 2: Accept (Write)**
    * **Goal:** Commit the value.
    * **Action:** Proposer sends `Prepare(n)`.
    * **Promise:** Acceptors promise to ignore any future request with $ID < n$. They also report the highest-numbered value they previously accepted.

2.  **Phase 2: Accept (Write)**
    * **Goal:** Commit the value.
    * **Action:** Proposer sends `Accept(n, v)`.
    * **Constraint:** If Phase 1 revealed an old value, the Proposer **must** adopt that value. If Phase 1 was clean, the Proposer can pick its own value.

## 3. The Critical Insight: Safety Rule (P2)
This ensures consistency even when the "fog of war" obscures whether a previous leader succeeded or failed.

* **The Rule:** *"If a proposal with value `v` is chosen, then every higher-numbered proposal that is chosen has value `v`."*
* **The "Fog of War" Logic:**
    * If Leader A crashes, Leader B sees only a partial picture (e.g., one node has data, others are empty).
    * Leader B cannot know if Leader A succeeded (wrote to a majority) or failed (wrote to a minority).
    * **The Fix:** To be safe, Leader B acts "paranoid." If it sees *any* trace of an old value, it assumes it *might* have won, and re-proposes (copies) it. This prevents overwriting a valid decision ("Phantom Write").
