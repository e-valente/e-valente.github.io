---
title: "Aray: Validating YARA Rules Without Handling Malware"
date: 2026-08-19T12:00:00-03:00
tags: ["detection-engineering", "yara", "malware-analysis", "ai-security", "software-engineering", "research", "blackhat"]
ShowToc: true
---

A YARA rule is easy to share. The malware sample used to prove that the rule works often is not.

That gap matters in detection engineering. Teams need positive fixtures to test scanners, alerting, incident-response automation, disaster-recovery workflows, and CI pipelines. Using live malware for those tests creates a separate operational problem: isolated infrastructure, handling controls, retention policy, and specialist oversight. A useful test artifact must satisfy the rule without reproducing the behavior of the malware that originally triggered it.

This was the starting point for **Aray**, an open-source project that synthesizes benign files which match YARA rules. I presented Aray at Black Hat USA 2026 Arsenal in Las Vegas.[1][2] Shortly afterwards, we published the research paper, *Aray: Deterministic-First Synthesis of Benign Artifacts for YARA Validation*.[1]

<p align="center">
  <img src="https://c2dc.github.io/aray/aray-logo.png" alt="Aray logo" width="220">
</p>

## A research collaboration with an engineering center of gravity

Aray grew from a collaboration among Lab-C2DC at the Aeronautics Institute of Technology (ITA), the University of São Paulo (USP), iFood, and Texas A&M University. The work brought together research questions from malware analysis and defensive AI with a practical software-engineering question: how can we turn a detection rule into a reliable, inspectable test fixture?[1][3]

The answer could not be "ask a model to generate a binary." A model-generated blob would be difficult to audit, hard to reproduce, and unreliable for byte-level constraints such as fixed offsets, integer reads, file-size limits, and executable-format headers. We instead treated the LLM as a constrained interpreter at the edge of a conventional construction system.

That architectural choice is the core of Aray: use probabilistic components only where interpretation is genuinely needed, then hand byte placement, layout, format construction, and verification to deterministic code.[1][3]

## From a YARA rule to a benign artifact

At a high level, Aray follows this pipeline:[1][3]

```text
YARA rule
  -> select a standalone target rule
  -> normalize and judge complex logic when necessary
  -> preflight constructibility
  -> extract fixed evidence into typed data
  -> construct an ELF, PE, or generic artifact
  -> verify the artifact with the YARA engine
```

![Overview of Aray turning a YARA rule into a benign synthesized artifact](https://c2dc.github.io/aray/diagrams/aray-overview.svg)

## Architecture: model-assisted interpretation, deterministic realization

The first stage is deliberately conservative. Aray reads the requested ruleset, selects the first non-private rule, and inlines only the helper rules reachable from that target. It assigns stable names to anonymous strings before any model call. This gives the rest of the pipeline a standalone, reviewable input rather than an ambiguous ruleset with unrelated rules.

Aray then checks whether the target already belongs to its constructible subset. Straightforward fixed strings, placements, simple ranges, count predicates, and supported integer comparisons can continue without semantic simplification. Regexes, partial hex wildcards, jumps, `or` conditions, and numeric `N of` expressions enter the normalization-and-judge path because they require a choice of witness or branch.[1][3]

The important boundary comes next. The normalized rule is parsed into typed string and constant entries. For supported constructs, that deterministic parse is authoritative: it supplies the literal bytes, encodings, exact offsets, ranges, multiplicity, integer width, and byte order used for construction. Structured model output may be retained for compatibility and observability, but it cannot override those fixed facts.[1][3]

![Aray architecture: interpretation, typed representation, deterministic backends, and YARA verification](https://c2dc.github.io/aray/diagrams/aray-architecture.svg)

Once the evidence is typed, Aray makes format and layout decisions with conventional code. Native MZ and PE-signature evidence route to the PE backend; ELF-compatible layouts route to the ELF backend; non-native magic at offset zero or layouts incompatible with executable headers route to a generic byte writer. The layout layer assigns non-colliding offsets, encodes ASCII, UTF-8, hex, and UTF-16LE witnesses as required, applies supported integer constants in the correct endian order, and enforces file-size constraints.[1][3]

The final backends are intentionally different from code generation by an LLM. The runnable ELF path emits GNU assembly and a linker script that make file offsets inspectable. The PE path uses MinGW where a regular executable layout is sufficient and a two-pass, offset-aware layout when a witness must land at a constrained file offset. `--scan-only` bypasses compilers entirely and writes minimal scanner-oriented ELF or PE structures directly. Generic artifacts remain raw byte blobs so their format magic and low-offset bytes are preserved.[3]

For the supported subset of YARA, Aray extracts strings, hex bytes, modifiers, offsets, ranges, match counts, and selected endian-aware integer checks deterministically. It then chooses an appropriate backend:

- **Linux ELF** artifacts can be assembled and linked with controlled section placement, or written directly in scan-only mode.
- **Windows PE** artifacts use MinGW-based construction where possible, with dedicated layout handling for offset-sensitive cases; scan-only generation can write a minimal PE structure directly.
- **Generic byte blobs** preserve non-executable format magic and exact low-offset layout without forcing an ELF or PE wrapper around the data.

The output is intentionally inspectable. Generated source, linker scripts, normalized rules, and intermediate artifacts are retained in the build directory, so an engineer can examine the construction rather than trust an opaque result.[1][3]

## A narrow role for LLMs

Some YARA rules are not immediately constructible. Regexes, wildcarded hex patterns, jump expressions, alternatives, and numeric `N of` conditions may require a simplification step before a concrete witness can be built.

Aray uses an LLM for that narrow task: propose a constructible normalization and, where required, judge the remaining semantic relationship. The model does not write C, assembly, linker scripts, PE headers, or arbitrary binary content. For supported fixed constructs, deterministic parsing is authoritative after the model call, which prevents omissions, escaping errors, incorrect offsets, and byte-order mistakes from reaching the backend.[1][3]

This boundary is important beyond this specific project. It separates a useful but fallible semantic component from the mechanisms that must be exact. In Aray, LangGraph coordinates the stages and retries; ordinary Python code, assemblers, linkers, and binary writers construct the result.[1][3]

## A concrete normalization example

The repository includes a sample based on a rule with a numeric `N of` condition. The original rule contains 24 string declarations; this shortened excerpt keeps only the relevant shape. The condition asks for any 15 of those strings, which is a valid YARA expression but not yet a concrete construction plan.[3]

```yara
rule malware_apt15_exchange_tool {
    strings:
        $s1 = "subjectname" fullword
        $s2 = "sendername" fullword
        $s3 = "WebCredentials" fullword
        // ... declarations through $s24 ...

    condition:
        uint16(0) == 0x5A4D and 15 of ($s*)
}
```

For a positive fixture, the normalizer can select one complete sufficient branch: require the PE `MZ` value and fifteen named witnesses. The full normalized rule retains the necessary declarations; the abbreviated condition below shows the transformation.

```yara
rule malware_apt15_exchange_tool {
    strings:
        $s1 = "subjectname" fullword
        $s2 = "sendername" fullword
        $s3 = "WebCredentials" fullword
        // ... retained declarations ...

    condition:
        uint16(0) == 0x5A4D and
        $s1 and $s2 and $s3 and $s4 and $s5 and
        $s6 and $s7 and $s8 and $s9 and $s10 and
        $s11 and $s12 and $s13 and $s14 and $s15
}
```

This is not an attempt to recreate the original malware. It is a constructive subset: any file satisfying the normalized condition satisfies the original `15 of ($s*)` predicate, while many files that satisfy the original rule will not satisfy this more specific selection. Aray treats that distinction as part of the contract and validates the generated result against the original rule with YARA.[1][3]

## Treating constructibility as an explicit outcome

Not every YARA rule can be synthesized safely or meaningfully. A fixed hash over an entire file is a preimage problem, not a layout problem. Conditions based on the YARA `pe` module require richer PE semantics than byte placement alone. Some integer equalities are simply impossible at the requested width.

Aray identifies these cases before artifact construction and records explicit dispositions such as unsupported, infeasible, or unsatisfiable. This is preferable to silently producing a file that looks plausible but does not test what the rule claims to detect.[1][3]

The final acceptance criterion is also external: a successful build is not enough. Aray invokes the real YARA engine, and a run counts as successful only when the generated benign artifact matches the source rule.[1][3]

## What the evaluation showed

The paper evaluates Aray on 416 public YARA rules. It reports that 406 rules passed constructibility preflight and that every generated fixture for those admitted rules matched its original upstream rule, for 406/416 overall and 406/406 among constructible rules.[1]

Those numbers do not mean that Aray implements all of YARA or that normalization is solved. They describe a concrete system result under a defined corpus and a strict end-to-end oracle. The paper also documents the remaining boundary: proving that every generated file preserves the full semantics of an arbitrary rule is a stronger problem than demonstrating one valid positive fixture.[1]

## Open source, research artifact, and next steps

We released Aray as an Apache-2.0 project with documentation, examples, evaluation tooling, and the generated artifacts needed to inspect its behavior.[3] The repository should be read as both software and a research artifact: it documents the supported YARA subset, the constructibility limits, the deterministic backends, and the evaluation methodology.[3]

For me, the project is also a useful pattern for building security tooling with LLMs. The goal was not to place a model in every stage of the pipeline. It was to constrain the model to a reviewable semantic role, build the critical path with conventional engineering, and validate the final result with the system that matters: YARA itself.

## Sources

[1] [Aray: Deterministic-First Synthesis of Benign Artifacts for YARA Validation](https://arxiv.org/abs/2608.19387)

[2] [Black Hat USA 2026 Arsenal schedule: ARAY](https://blackhat.com/us-26/arsenal/schedule/#aray-benign-binary-synthesis-for-signature-validation-without-the-malware-52919)

[3] [Aray source code and documentation](https://github.com/c2dc/aray)
