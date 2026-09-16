# Kickoff prompt for a frontend specialization

Use this prompt with the available model description, target documentation and
formal definitions, existing code, and deployment requirements. It produces a
design for a specialized interface; it does not assume one universal language
or automatically establish any semantic claim.

---

Design a VeriTac frontend specialization for the supplied computation and target.
Use `docs/frontend_design.md` as the architectural contract. Design the interface
yourself before delegating implementation, unless explicitly directed otherwise.

Start by inspecting the supplied artifacts and existing interfaces. Identify
the intended deliverable and the authoritative source of the computation's
meaning. Distinguish implemented capabilities, proven claims, declared
assumptions, empirical observations, and proposals. Cite the relevant sources.
Do not infer ISA semantics from performance tables or examples alone.

Address these concerns in the vocabulary appropriate to this specialization:

- Observable behavior, input domain, persistent state, and external interaction.
- Numerical semantics and the reference against which correctness or error is
  measured, including exceptional cases and any approximation budget.
- Execution model, memory/layout/ownership, ordering, completion, capabilities,
  and runtime or deployment assumptions.
- Permitted transformations, fixed requirements, and optimization objectives.
- Representations, interpretation boundaries, proof relations, and acceptance
  evidence, from the source input through the requested output artifact.

For each concern, give its precise representation or identify an unresolved
boundary. Explain omissions when a concern does not apply. Ask focused questions
where an answer would change meaning or acceptance; continue independent design
work and label provisional assumptions. Do not silently weaken the task.

Choose native input syntax and intermediate views that make this workload and
target easy to express and verify. Reuse existing representations when their
semantics fit. Do not force the design into a universal tensor/loop/instruction
schema. Describe the minimal references and claim/evidence metadata needed by
shared orchestration separately from the specialized payloads.

Produce a concise design dossier containing:

1. The specialization's scope and key decisions, with alternatives only where
   there is a material tradeoff.
2. A computation contract, target contract, representation/proof adapters, and
   optimization request. Classify constraints by their role and provenance.
3. One concrete user input, its interpretation, a useful AI proposal/edit, the
   resulting proof obligations, and the expected acceptance artifacts.
4. At least one semantic rejection and one resource/ordering rejection when
   applicable. Show how diagnostics support repair without changing the fixed
   contract. Describe unsupported inputs and proof timeouts separately.
5. A map of reusable proofs, new proof obligations, and remaining trust
   boundaries, including importers, compilers, loaders, and physical conformance.
6. The smallest implementation slice that demonstrates the interface and its
   stated acceptance claim, with replay and negative checks.

Keep performance measurements separate from correctness evidence. Bind claims
to actual artifacts and semantic versions. If the complete requested proof path
is unavailable, show precisely where it stops. A readable prompt, successful
test, or certificate hash is not a substitute for a checked semantic connection.
