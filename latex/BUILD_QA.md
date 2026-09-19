# Build and layout QA

检查日期：2026-09-10。

- 编译环境：Conda `latex`；`latexmk 4.88`、pdfTeX/TeX Live 2026、BibTeX。
- 编译命令：`conda run --no-capture-output -n latex latexmk -pdf main.tex`。
- 编译结果：成功生成 `main.pdf`，7 页，US Letter（612 x 792 pt），约 292 KB。
- 文献：21 条 BibTeX 记录，BibTeX 正常完成；无未定义 citation/reference。
- 版式日志：无 `Overfull \\hbox`、`Overfull \\vbox` 或 float 警告。仅有窄双栏长术语、超参数列表和红色 TODO 造成的非致命 `Underfull \\hbox`。
- 字体：`pdffonts` 检查显示全部为嵌入、子集化的 Type 1 字体。
- 页面渲染：使用 `pdftoppm` 渲染并目检全部 7 页；题目、摘要、TikZ 方法总览、公式、双栏正文、跨栏表格、参考文献均可读，无裁切或重叠。
- 最后一页：当前不手工平衡参考文献栏。实验结果与图表加入后分页会变化，投稿前再执行 camera-ready 的最终平衡。
- 数据完整性：共 10 处红色 `\\todo{}` 与主表/消融表数值占位符；没有把未完成实验写成结果。
