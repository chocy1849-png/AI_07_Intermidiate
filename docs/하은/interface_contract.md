# Interface Contract

## 목적

Track A의 RAG 처리 함수와 Track B의 Streamlit UI가 바로 연결될 수 있도록 입출력 규격을 통일한다.

## 1. 문서 파싱 인터페이스

```python
parse_document(file_path: str | Path) -> dict
```

반환 포맷:

```python
{
    "text": "문서 전체 텍스트",
    "metadata": {
        "source_path": "/abs/path/to/file.hwp",
        "file_name": "sample.hwp",
        "file_format": "hwp",
        "parser": "olefile-hwp5",
        "page_count": None,
        "char_count": 12345,
    }
}
```

## 2. 청킹 인터페이스

```python
chunk_document(
    text: str,
    metadata: dict | None = None,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]
```

반환 포맷:

```python
[
    {
        "chunk_id": "sample.hwp::0",
        "text": "청크 텍스트",
        "metadata": {
            "file_name": "sample.hwp",
            "chunk_index": 0,
            "chunk_size": 1000,
            "overlap": 200,
        },
    }
]
```

## 3. 검색 인터페이스

```python
retrieve(
    query: str,
    filters: dict | None = None,
    top_k: int = 5,
) -> list[dict]
```

`filters` 예시:

```python
{
    "agency": "한국수자원공사",
    "keyword": "ERP",
}
```

## 4. 응답 생성 인터페이스

```python
generate_answer(
    query: str,
    retrieved_docs: list[dict],
    chat_history: list[dict] | None = None,
) -> dict
```

반환 포맷:

```python
{
    "answer": "최종 답변",
    "citations": [
        {
            "file_name": "sample.hwp",
            "chunk_id": "sample.hwp::0",
        }
    ],
    "debug": {
        "retrieved_count": 5,
        "model": "dummy-v1",
    },
}
```

## 5. UI 연동 규칙

- Streamlit Chat 화면은 `generate_answer()` 반환값의 `answer`를 그대로 출력한다.
- 인용 정보가 있을 때는 `citations`를 파일명 기준으로 표시한다.
- 실제 LLM 연결 전에는 동일한 키 구조를 유지한 더미 응답을 사용한다.

