# CodeSage

这个文档提供了一个完整的 Python 脚本，演示了如何利用 LlamaIndex 的 CodeSplitter 进行 AST 代码切割，结合 Qwen3 兼容的 Embedding 模型进行向量化，并将结果存储在 LanceDB 中以实现高效的自然语言代码召回。

如何运行此示例：

安装依赖：

pip install llama-index-core
pip install llama-index-embeddings-huggingface # 用于 Qwen3 兼容的 Embedding 模型
pip install llama-index-vector-stores-lancedb # 用于 LanceDB 集成
pip install lancedb  # LanceDB 库本身
pip install transformers  # Hugging Face 模型需要
pip install tree-sitter tree-sitter-languages # CodeSplitter 需要
pip install pandas # 示例代码中使用了 pandas，用于确保环境完整

保存代码： 将上面的 Python 代码保存为一个 .py 文件，例如 code_retrieval_demo.py。

运行脚本： 在终端中导航到保存文件的目录，然后运行：

python code_retrieval_demo.py

脚本会自动创建示例代码文件，构建 LanceDB 数据库，执行查询，并清理数据。

后续可能的改进和探索：

更复杂的代码库： 将此方案应用于你自己的大型代码库。

性能优化： 针对生产环境，进一步优化 CodeSplitter 的参数（chunk_lines, chunk_lines_overlap, max_chars），以及 LanceDB 的配置。

多语言支持： CodeSplitter 支持多种编程语言。你可以扩展示例以包含 Java, JavaScript 等其他语言的代码。

用户界面： 构建一个简单的 Web 界面（例如使用 Streamlit 或 Flask），让用户可以通过浏览器输入查询并查看结果。

反馈循环： 收集用户对检索结果的反馈，用于持续改进嵌入模型或召回策略。

混合召回： 结合关键词搜索和向量相似度搜索，以获得更全面的结果。

专用 Embedding 模型： 持续关注 Qwen 系列是否有专门针对代码或中文代码的更优 Embedding 模型，并进行替换和测试。

希望这个文档对你有所帮助！