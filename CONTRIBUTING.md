# Contributing Guide

感谢你对本项目的兴趣. 本指南面向希望本地运行、调试或扩展该信用风险评分项目的开发者.

## 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/MeaFew/riskscore.git
cd riskscore

# 2. 创建虚拟环境 (推荐 Python 3.12)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 数据准备

项目使用 Kaggle Home Credit Default Risk 数据集. 请运行下载脚本获取数据集:

```bash
bash download_data.sh
```

## 本地工作流

```bash
# 1. 数据预处理
make preprocess

# 2. 特征工程 (WOE/IV + target encoding)
make features

# 3. 模型训练 (LR/RF/XGB/LGBM)
make train

# 4. 模型评估 (AUC/KS/Gini)
make evaluate

# 5. SHAP 可解释性分析
make shap

# 6. 启动看板
make dashboard
```

## 代码规范

提交前请确保通过以下检查:

```bash
# Python lint (ignores are centralised in pyproject.toml)
ruff check scripts/ tests/ dashboard/

# 单元测试
pytest tests/ -v
```

## 脚本导入约定

`scripts/` 已打包为 Python package (`scripts/__init__.py`)。脚本既可以作为 `__main__` 直接运行 (`python scripts/foo.py`)，也可以作为模块运行 (`python -m scripts.foo`)。`__main__` 模块无法使用相对导入，因此每个 CLI 脚本顶部保留一次 `sys.path.insert(0, project_root)`，随后通过 `scripts.` 命名空间导入同级模块，例如:

```python
from scripts.metrics_utils import ks_score
from scripts.train_models import make_pipeline
```

请勿恢复旧的顶层 `from metrics_utils import ...` 写法，以免破坏 package 导入的一致性。

## 提交规范

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `refactor:` 重构
- `ci:` 持续集成相关
- `test:` 测试相关

## 扩展建议

- 新增模型: 在 `scripts/train_models.py` 中添加
- 新增特征: 在 `scripts/feature_engineering.py` 中扩展 WOE/IV 逻辑
- 新增分析: 放在 `scripts/` 并更新 Makefile
