"""Generate/verify the standalone knowledge contract; optionally vendor a consumer.

python tools/sync_knowledge_contract.py [--check] [--consumer ../xgen-omnifuse]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import sys
import types
import typing

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/knowledge/v1"


def artifacts():
    spec = importlib.util.spec_from_file_location("knowledge_records", CONTRACT / "records.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    records = [module.Resource, module.SourceChunk, module.EmbeddingProfile, module.ChunkEmbedding,
               module.Evidence, module.GraphEntity, module.GraphFact, module.KnowledgeBundle]

    def schema_type(hint):
        if hint is typing.Any:
            return {}
        simple = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null", dict: "object"}
        if hint in simple:
            return {"type": simple[hint]}
        origin, args = typing.get_origin(hint), typing.get_args(hint)
        if origin in (types.UnionType, typing.Union):
            return {"anyOf": [schema_type(arg) for arg in args]}
        if origin in (tuple, list):
            return {"type": "array", "items": schema_type(args[0])}
        return {"$ref": f"#/$defs/{hint.__name__}"}

    definitions = {}
    for cls in records:
        hints = typing.get_type_hints(cls)
        required = [f.name for f in dataclasses.fields(cls)
                    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
        definitions[cls.__name__] = {
            "type": "object", "additionalProperties": False,
            "properties": {name: schema_type(hint) for name, hint in hints.items()}, "required": required,
        }
    definitions["KnowledgeBundle"]["properties"]["contract_version"] = {"const": module.CONTRACT_VERSION}
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "urn:xgen:knowledge:1.0",
              "$ref": "#/$defs/KnowledgeBundle", "$defs": definitions,
              "$comment": "Structural schema plus records.py semantic validation (IDs, source revisions, cycles, vectors)."}
    code = (CONTRACT / "records.py").read_bytes()
    schema_bytes = (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode()
    lock = {"contract_version": module.CONTRACT_VERSION,
            "records_sha256": hashlib.sha256(code).hexdigest(),
            "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
            "source": "PlateerLab/xgen-ontology-build/contracts/knowledge/v1"}
    lock_bytes = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode()
    return {"knowledge.py": code, "knowledge.schema.json": schema_bytes, "knowledge.lock.json": lock_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--consumer", type=pathlib.Path)
    args = parser.parse_args()
    data = artifacts()
    targets = {ROOT / "src/xgen_ontology" / name: value for name, value in data.items()}
    targets[CONTRACT / "schema.json"] = data["knowledge.schema.json"]
    targets[CONTRACT / "lock.json"] = data["knowledge.lock.json"]
    if args.consumer:
        targets.update({args.consumer / "src/omnifuse" / name: value for name, value in data.items()})
    for path, value in targets.items():
        if args.check:
            if not path.exists() or path.read_bytes() != value:
                raise SystemExit(f"contract drift: {path}")
        else:
            path.write_bytes(value)
    print("knowledge/v1 contract verified" if args.check else "knowledge/v1 artifacts generated")


if __name__ == "__main__":
    main()
