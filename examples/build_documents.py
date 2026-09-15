"""Build an ontology from documents with zero LLM calls, then search it.

The base build reads document *structure*: tables become row entities with
their column values as attributes and relations, prose yields noun-phrase
entities, and the hierarchy comes from Hearst patterns plus the structure of the
names themselves. Install the ``korean`` extra for the prose parts; tables need
nothing.
"""
import sys

sys.path.insert(0, "src")

from xgen_ontology import build_from_documents  # noqa: E402

BUDGET = """예산 배정표
<table>
<tr><td>기관명</td><td>담당부서</td><td>예산</td></tr>
<tr><td>한국마사회</td><td>말산업연구소</td><td>12억원</td></tr>
<tr><td>농림축산식품부</td><td>축산정책과</td><td>30억원</td></tr>
<tr><td>제주특별자치도</td><td>축산과</td><td>5억원</td></tr>
</table>
"""
# A document may be given pre-chunked (a list of passages); the hierarchy pass judges a
# hypernym's discriminativeness across passages, so give it a few.
NOTES = [
    "상임감사실은 내부감사를 맡는다. 준법감사실은 법규 준수를 본다. 감사실은 두 부서를 아우른다.",
    "혈액, 모근 등의 생체시료는 검사 전에 보관된다.",
    "모바일, 인터넷 등 대체채널로도 신청할 수 있다.",
    "검사 결과는 의뢰 기관에 통보된다.",
    "보관 기간이 지난 시료는 폐기한다.",
    "대체채널 신청은 본인 인증을 거친다.",
    "내부감사 결과는 이사회에 보고한다.",
]

onto = build_from_documents({"budget.md": BUDGET, "notes.md": NOTES}, chunk=False)   # mode="basic" (default)

print("mode / llm calls :", onto.report.mode, onto.report.llm_calls)
print("classes          :", [c.name for c in onto.concepts.classes][:8])
print("hierarchy        :", onto.concepts.class_hierarchy[:6])
print("instances        :", [(i.name, i.class_name) for i in onto.instances][:6])
print("relations        :", [(r.subject, r.predicate, r.object) for r in onto.relations][:4])
print("data values      :", [(d.entity, d.property, d.value) for d in onto.data_values][:4])
print("quality          :", onto.report.quality["score"], onto.report.quality["warnings"])

res = onto.search("한국마사회 예산")
print("\nevidence chunks  :", [c.id for c in res.chunks])
print("relations used   :", res.relations[:3])

# mode="enrich" adds an LLM pass for relations between these entities, mode="llm" is full
# LLM extraction: build_from_documents(docs, llm=CallableLLM(my_model), mode="enrich")
