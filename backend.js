import express from "express";
import bodyParser from "body-parser";
import Groq from "groq-sdk";

const app = express();
const port = 3000;

app.use(bodyParser.json());

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// /api/extract - Extract code and question name
app.post("/api/extract", (req, res) => {
  const { question, code } = req.body;
  if (!question || !code) {
    return res.status(400).json({ error: "Missing question or code" });
  }
  res.json({ questionName: question, extractedCode: code });
});

// /api/generate - Generate code using Groq API
app.post("/api/generate", async (req, res) => {
  const { prompt } = req.body;
  if (!prompt) {
    return res.status(400).json({ error: "Missing prompt" });
  }
  try {
    const response = await groq.chat.completions.create({
      messages: [{ role: "user", content: `Generate code for: ${prompt}` }],
      model: "llama-3.3-70b-versatile",
    });
    res.json({ generatedCode: response.choices[0]?.message?.content || "" });
  } catch (error) {
    res.status(500).json({ error: "Error generating code", details: error.message });
  }
});

// /api/debug - Debug code using Groq API
app.post("/api/debug", async (req, res) => {
  const { code } = req.body;
  if (!code) {
    return res.status(400).json({ error: "Missing code" });
  }
  try {
    const debugPrompt = `Debug the following code and provide suggestions:\n${code}`;
    const response = await groq.chat.completions.create({
      messages: [{ role: "user", content: debugPrompt }],
      model: "llama-3.3-70b-versatile",
    });
    res.json({ debuggedOutput: response.choices[0]?.message?.content || "" });
  } catch (error) {
    res.status(500).json({ error: "Error debugging code", details: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});