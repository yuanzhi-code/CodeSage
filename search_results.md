# 代码搜索结果

## 结果 1

### 相似度
0.6017

### 代码片段
```
def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/agent/graph.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

## 结果 2

### 相似度
0.1525

### 代码片段
```
def find_topics(self, search_term: str = None, image: str = None, top_n: int = 5) -> Tuple[List[int], List[float]]:
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/bertopic/_bertopic.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

## 结果 3

### 相似度
0.1110

### 代码片段
```
def query_code(self, query: str, similarity_top_k: int = 10) -> Dict[str, Any]:
        """
        查询代码。

        Args:
            query: 查询文本
            similarity_top_k: 初步召回的结果数量

        Returns:
            Dict[str, Any]: 查询结果，包含生成的摘要和源代码片段
        """
        if not self.index:
            raise ValueError("请先调用 process_code_repo() 处理代码仓库")

        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            node_postprocessors=[self.reranker]
        )

        response = query_engine.query(query)
        
        return {
            "summary": response.response,
            "source_nodes": [
                {
                    "file_path": node.metadata.get('file_path', '未知文件'),
                    "score": node.score,
                    "content": node.text
                }
                for node in response.source_nodes
            ]
        }

    def cleanup(self) -> None:
        """清理 LanceDB 数据库。"""
        if os.path.exists(self.lancedb_uri):
            import shutil
            shutil.rmtree(self.lancedb_uri)
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/code_processor.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

## 结果 4

### 相似度
0.0840

### 代码片段
```
def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/agent/graph.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

## 结果 5

### 相似度
0.0381

### 代码片段
```
def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/agent/graph.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

## 结果 6

### 相似度
0.0364

### 代码片段
```
from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/agent/tools_and_schemas.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

## 结果 7

### 相似度
0.0216

### 代码片段
```
from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/agent/utils.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

## 结果 8

### 相似度
0.0129

### 代码片段
```
# 在请求之间添加延时，避免请求过于频繁
        time.sleep(2)
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/src/rss/RssReader.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

## 结果 9

### 相似度
0.0112

### 代码片段
```
"""LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/agent/graph.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

## 结果 10

### 相似度
0.0061

### 代码片段
```
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}
```

### 元数据
```json
{'file_path': '/Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/agent/graph.py', 'start_line': 1, 'type': 'CodeBlock'}
```

---

