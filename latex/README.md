# DistriSurg IEEE/ICRA LaTeX draft

该目录是与 `../distrisurg/` 研究代码对齐的 IEEE 双栏论文草稿。

- `main.tex`：完整论文骨架；Introduction、Related Work 和 Method 已写好。
- `references.bib`：经 CVF、MICCAI、PubMed/出版社或 arXiv 页面核对的参考文献。
- `ieeeconf.cls`：复用参考工程的 IEEE robotics conference 模板。
- `AUTHOR_CHECKLIST_ZH.md`：术语表、claim--evidence 对照及全部实验 TODO。
- `BUILD_QA.md`：编译、日志和页面渲染检查记录。
- `main.pdf`：使用 Conda `latex` 环境生成的 PDF。

## 编译

```bash
conda activate latex
cd /home/data/mashixing/dataset_18tb/icra-2027/latex
latexmk -pdf main.tex
```

也可以不激活环境：

```bash
conda run --no-capture-output -n latex latexmk -pdf main.tex
```

清理辅助文件但保留 PDF：

```bash
conda run --no-capture-output -n latex latexmk -c main.tex
```

## 写作约定

所有红色 `[TODO: ...]` 都表示需要作者元数据、冻结后的实验结果或真实失败分析。主表中的红色 `--` 是数值占位符，不是零值。完成实验前请勿删除这些标记，也不要将旧版 LaMa/LCM 先导结果冒充 DistriSurg 正式结果。

