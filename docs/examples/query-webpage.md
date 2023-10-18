```python
from zeta.artifacts import BaseArtifact
from zeta.drivers import LocalVectorStoreDriver
from zeta.loaders import WebLoader


vector_store = LocalVectorStoreDriver()

[
    vector_store.upsert_text_artifact(a, namespace="zeta")
    for a in WebLoader(max_tokens=100).load("https://www.zeta.ai")
]

results = vector_store.query(
    "creativity",
    count=3,
    namespace="zeta"
)

values = [BaseArtifact.from_json(r.meta["artifact"]).value for r in results]

print("\n\n".join(values))
```