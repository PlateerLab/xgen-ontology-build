"""Build-only example: python examples/knowledge_build.py /tmp/knowledge.json."""
import sys

from xgen_ontology import build_knowledge
from xgen_ontology.knowledge import KnowledgeBundle, Resource, SourceChunk

source = KnowledgeBundle(
    corpus_id="example", snapshot_id="source-v1",
    resources=(Resource("root", "v1", "policies", "directory"),
               Resource("colors", "v1", "colors.csv", parent_id="root")),
    chunks=(SourceChunk("colors:v1:0", "colors", "v1", "csv:v1", "id,name\n1,Red\n2,Blue",
                        locator={}),),
)
built = build_knowledge(source, snapshot_id="graph-v1")
built.dump(sys.argv[1])
print(f"Built {len(built.entities)} entities, {len(built.facts)} facts")
