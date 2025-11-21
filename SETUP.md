# 设置远程仓库

## 1. 在 GitHub/Gitee 等平台创建仓库

在 GitHub、Gitee 或其他 Git 托管平台创建一个新仓库。

## 2. 添加远程仓库

```bash
# GitHub 示例
git remote add origin https://github.com/your-username/gemini-webapi.git

# 或使用 SSH
git remote add origin git@github.com:your-username/gemini-webapi.git
```

## 3. 推送代码

```bash
# 推送到 main 分支
git push -u origin main

# 或推送到 master 分支（如果使用 master）
git push -u origin master
```

## 4. 验证

访问你的远程仓库，确认代码已成功上传。

## 注意事项

- `config.json` 文件已添加到 `.gitignore`，不会被上传
- 确保 `config.json.example` 已提交，供其他用户参考
- 敏感信息（Cookie）不会泄露到仓库中

