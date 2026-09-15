# knowledge/v1 canonical contract

- `records.py`: canonical zero-dependency records and semantic validation.
- `schema.json`: generated, language-neutral structural JSON Schema.
- `lock.json`: version and content digests.
- [Contract semantics and integration guide](../../../docs/knowledge.md).

Generate and verify all copies with `tools/sync_knowledge_contract.py` from the
repository root. Consumers vendor the generated artifacts; neither package imports
the other at runtime.
