import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from openai import OpenAI

app = FastAPI()

# 初始化 AI 客户端（填入你的 DeepSeek Key）
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

# ================= 数据库部分 =================
def init_db():
    conn = sqlite3.connect("notes.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

class NoteIn(BaseModel):
    content: str

class SummarizeIn(BaseModel):
    note_id: int

@app.post("/notes")
def create_note(note: NoteIn):
    conn = sqlite3.connect("notes.db")
    cur = conn.execute("INSERT INTO notes (content) VALUES (?)", (note.content,))
    conn.commit()
    note_id = cur.lastrowid
    conn.close()
    return {"status": "成功", "id": note_id}

@app.get("/notes")
def get_notes():
    conn = sqlite3.connect("notes.db")
    rows = conn.execute("SELECT id, content FROM notes").fetchall()
    conn.close()
    return [{"id": r[0], "content": r[1]} for r in rows]

# ================= AI 总结接口 =================
@app.post("/summarize")
def summarize_note(data: SummarizeIn):
    # 1. 先去数据库里找这篇笔记
    conn = sqlite3.connect("notes.db")
    row = conn.execute("SELECT content FROM notes WHERE id = ?", (data.note_id,)).fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="没找到这篇笔记")
    
    note_content = row[0]
    
    # 2. 把笔记发给 DeepSeek，让它总结
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的总结助手。请用一句话精简地总结用户提供的笔记内容。"},
                {"role": "user", "content": note_content}
            ]
        )
        summary = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 调用失败: {str(e)}")
        
    # 3. 把总结返回去
    return {"note_id": data.note_id, "original": note_content, "ai_summary": summary}
# ================= AI 自主调用（Function Calling） =================
class UserInput(BaseModel):
    text: str

@app.post("/agent")
def ai_agent(data: UserInput):
    # 1. 告诉 AI，你现在有哪些“工具”可以使用
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_note",
                "description": "当用户想要记录、保存某段内容时，调用此工具存入数据库。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "要保存的笔记内容"}
                    },
                    "required": ["content"]
                }
            }
        }
    ]

    # 2. 发送用户的输入给 AI，并附上工具清单
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个个人知识库助手，帮用户记录东西。"},
                {"role": "user", "content": data.text}
            ],
            tools=tools  # 把工具交给 AI
        )
        
        message = response.choices[0].message

        # 3. AI 决定调用工具了！
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            # 解析 AI 想存的内容
            import json
            args = json.loads(tool_call.function.arguments)
            note_content = args.get("content")
            
            # 4. 你的代码帮 AI 真正执行“存数据库”的操作
            conn = sqlite3.connect("notes.db")
            cur = conn.execute("INSERT INTO notes (content) VALUES (?)", (note_content,))
            conn.commit()
            note_id = cur.lastrowid
            conn.close()
            
            return {
                "reply": f"好的，我已经为您记录了笔记（ID: {note_id}）。",
                "ai_action": "自动调用了存笔记接口",
                "saved_content": note_content
            }
        else:
            # AI 觉得不需要调工具，正常聊天
            return {"reply": message.content}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))