from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_template(
    "Classify this review as POSITIVE, NEUTRAL, or NEGATIVE in one word:\n{review}"
)

llm = ChatOllama(model="llama3.2", temperature=0)
chain = prompt | llm

reviews = [
    {"review": "App fast hai aur dark mode mast kaam kar raha hai."},
    {"review": "Payment kat gaya par order update nahi hua."},
    {"review": "UI thoda simple hai, bas chal raha hai."},
    {"review": "Customer support ne 10 min mein refund process kar diya."},
    {"review": "Login karte waqt crash aa raha hai."},
]


def process_reviews_at_scale(items: list[dict], concurrency: int = 2) -> list[str]:
    try:
        # Ek saath sirf `concurrency` requests chalengi
        responses = chain.batch(items, config={"max_concurrency": concurrency})
        return [res.content.strip() for res in responses]
    except Exception as exc:
        # Agar batch level par koi issue aaye toh item-by-item safe fallback
        results = []
        for item in items:
            try:
                res = chain.invoke(item)
                results.append(res.content.strip())
            except Exception as item_exc:
                results.append(f"FAILED: {item_exc}")
        return results


labels = process_reviews_at_scale(reviews, concurrency=2)
for item, label in zip(reviews, labels):
    print(f"{label:10} | {item['review']}")