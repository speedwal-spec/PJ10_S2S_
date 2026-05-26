# Git 推送消融实验成果到 GitHub 分支 - 完整指南

## 🎯 目标

将消融实验的代码、脚本、文档推送到 GitHub 的新分支 `ablation-experiment`

---

## 📋 推送前的准备

### 1. 检查当前状态

```powershell
# 查看当前分支和修改状态
git status

# 查看远程仓库配置
git remote -v
```

**预期输出**：
- 当前在 `dev` 分支
- 有多个修改的文件和未跟踪的新文件
- 远程仓库：`origin git@github.com:runtangtang/PJ10.git`

---

## 🚀 推送步骤（按顺序执行）

### 步骤 1：创建新分支

```powershell
# 从当前 dev 分支创建并切换到 ablation-experiment 分支
git checkout -b ablation-experiment
```

**说明**：
- `checkout -b` = 创建新分支 + 切换到该分支
- 新分支会包含当前 dev 分支的所有代码
- 你的修改会保留在这个新分支上

**验证**：
```powershell
# 确认已切换到新分支
git branch
```
应该看到 `* ablation-experiment`（星号表示当前分支）

---

### 步骤 2：添加文件到暂存区

#### 方案 A：添加所有文件（推荐）

```powershell
# 添加所有修改和新文件
git add .
```

#### 方案 B：选择性添加（更精确）

```powershell
# 只添加消融实验相关的文件
git add train.py
git add predict.py
git add prepare_data.py
git add data_manifest.json

# 添加实验脚本
git add run_ablation_experiments.ps1
git add run_baseline_quick.ps1
git add install_dependencies.ps1

# 添加分析工具
git add extract_results.py
git add generate_comparison_plots.py

# 添加文档
git add "实验记录表.md"
git add "消融实验分析报告.md"
git add "演示文稿大纲.md"
git add "TODO_消融实验阶段.md"
git add "可视化分析指南.md"
```

**注意**：
- ✅ **会被添加**：代码、脚本、文档、配置文件
- ❌ **不会被添加**（已在.gitignore中）：
  - `t5-news-checkpoint/` （模型文件，太大）
  - `data_cache/` （数据缓存）
  - `__pycache__/` （Python缓存）

---

### 步骤 3：提交更改

```powershell
# 提交到本地仓库
git commit -m "feat: 完成T5新闻摘要消融实验

- 实现10组系统性对照实验（学习率/Batch Size/序列长度/数据量）
- 添加自动化实验脚本 (run_ablation_experiments.ps1)
- 生成完整实验报告和分析文档
- 建立基准配置和优化建议
- ROUGE-L提升最高达9.6%（lr=0.001）
- 验证数据量Scaling Law（40→120样本，+64%性能）

详细结果见：消融实验分析报告.md"
```

**提交信息规范**：
- 第一行：简短描述（50字符以内）
- 空一行
- 详细说明：列出主要改动点
- 使用中文或英文均可

---

### 步骤 4：推送到 GitHub

```powershell
# 首次推送新分支到远程
git push -u origin ablation-experiment
```

**参数说明**：
- `-u` = `--set-upstream`，设置上游分支
- 之后只需 `git push` 即可

**首次推送可能需要认证**：
- 如果使用 SSH（`git@github.com`），确保SSH密钥已配置
- 如果使用 HTTPS，可能需要输入GitHub用户名和密码/token

---

### 步骤 5：验证推送成功

在浏览器中访问：
```
https://github.com/runtangtang/PJ10/tree/ablation-experiment
```

应该能看到：
- ✅ 新分支 `ablation-experiment`
- ✅ 所有提交的代码和文档
- ✅ 最新的 commit 信息

---

## 🔧 常见问题解决

### 问题 1：推送被拒绝（rejected）

**错误信息**：
```
! [rejected] ablation-experiment -> ablation-experiment (fetch first)
```

**解决方案**：
```powershell
# 先拉取远程更新
git pull origin ablation-experiment

# 如果有冲突，解决后再推送
git push -u origin ablation-experiment
```

---

### 问题 2：文件太大无法推送

**错误信息**：
```
remote: error: File xxx is 1xx MB; this exceeds GitHub's file size limit of 100.00 MB
```

**解决方案**：
```powershell
# 1. 从暂存区移除大文件
git rm --cached t5-news-checkpoint/baseline/model.safetensors

# 2. 确认 .gitignore 已包含该目录
# t5-news-checkpoint/ 应该在 .gitignore 中

# 3. 重新提交
git commit -m "fix: 移除大模型文件"

# 4. 再次推送
git push -u origin ablation-experiment
```

---

### 问题 3：SSH密钥未配置

**错误信息**：
```
git@github.com: Permission denied (publickey).
```

**解决方案 A：配置SSH密钥**
```powershell
# 1. 生成SSH密钥（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 复制公钥
cat ~/.ssh/id_ed25519.pub
# 或 Windows: type $env:USERPROFILE\.ssh\id_ed25519.pub

# 3. 添加到 GitHub
# 访问: https://github.com/settings/keys
# 点击 "New SSH key"，粘贴公钥

# 4. 测试连接
ssh -T git@github.com
```

**解决方案 B：改用HTTPS**
```powershell
# 修改远程URL为HTTPS
git remote set-url origin https://github.com/runtangtang/PJ10.git

# 然后推送（需要输入GitHub账号密码或token）
git push -u origin ablation-experiment
```

---

### 问题 4：想修改最后一次提交

```powershell
# 修改提交信息
git commit --amend -m "新的提交信息"

# 添加遗漏的文件
git add 遗漏的文件.txt
git commit --amend --no-edit

# 如果已经推送，需要强制推送
git push -f origin ablation-experiment
```

**⚠️ 警告**：`-f` 强制推送会覆盖远程历史，仅在个人分支使用！

---

## 📊 推送后的操作

### 1. 创建 Pull Request（可选）

如果想合并到主分支：

1. 访问：`https://github.com/runtangtang/PJ10/pulls`
2. 点击 "New pull request"
3. 选择：
   - base: `main` 或 `dev`
   - compare: `ablation-experiment`
4. 填写PR描述
5. 点击 "Create pull request"

### 2. 继续开发

在新分支上可以继续工作：

```powershell
# 修改文件后
git add .
git commit -m "fix: 修复xxx问题"
git push  # 不需要 -u，已经设置过上游分支
```

### 3. 切换回原分支

```powershell
# 切换回 dev 分支
git checkout dev

# 查看所有分支
git branch -a
```

---

## 🎨 分支管理最佳实践

### 分支命名规范

```
feature/xxx     - 新功能
bugfix/xxx      - bug修复
experiment/xxx  - 实验性代码
ablation-xxx    - 消融实验
```

### 推荐的分支策略

```
main (生产环境)
  └─ dev (开发环境)
      ├─ ablation-experiment (当前分支)
      ├─ feature/new-model
      └─ bugfix/fix-dataloader
```

---

## 📝 完整命令清单（快速复制）

```powershell
# 1. 进入项目目录
cd E:\develop\PJ\PJ10_S2S_

# 2. 创建新分支
git checkout -b ablation-experiment

# 3. 添加所有文件
git add .

# 4. 提交
git commit -m "feat: 完成T5新闻摘要消融实验

- 实现10组系统性对照实验
- 添加自动化实验脚本
- 生成完整实验报告和分析文档
- ROUGE-L提升最高达9.6%"

# 5. 推送到GitHub
git push -u origin ablation-experiment

# 6. 验证
git status
```

---

## ✅ 检查清单

推送前确认：

- [ ] 已创建新分支 `ablation-experiment`
- [ ] `.gitignore` 已排除大文件
- [ ] 所有重要代码和文档已添加
- [ ] 提交信息清晰明了
- [ ] SSH密钥已配置或改用HTTPS
- [ ] 网络连接正常

推送后确认：

- [ ] GitHub上能看到新分支
- [ ] 所有文件都已上传
- [ ] Commit信息显示正确
- [ ] 可以在线查看文档内容

---

## 💡 小贴士

1. **定期推送**：不要等所有工作做完再推送，每完成一个阶段就推送一次

2. **原子提交**：每次提交只做一件事，便于追溯和管理

3. **分支保护**：可以在GitHub设置分支保护规则，防止误删

4. **清理分支**：实验结束后，可以删除远程分支
   ```powershell
   git push origin --delete ablation-experiment
   ```

5. **查看历史**：
   ```powershell
   # 查看提交历史
   git log --oneline
   
   # 查看图形化历史
   git log --graph --oneline --all
   ```

---

## 🎉 完成！

按照以上步骤，你就可以成功将消融实验成果推送到GitHub了！

**下一步**：
1. 在GitHub上创建README说明实验内容
2. 邀请团队成员review代码
3. 考虑是否合并到主分支
4. 开始下一阶段的优化工作

祝你推送顺利！🚀
