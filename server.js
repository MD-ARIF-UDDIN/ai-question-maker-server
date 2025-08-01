require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const Tesseract = require('tesseract.js');
const fs = require('fs');
const path = require('path');

const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
app.use(cors());
app.use(express.json());

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

const upload = multer({ storage: multer.memoryStorage() });

app.post('/api/upload-file', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

        const mimeType = req.file.mimetype;
        let extractedText = '';

        if (mimeType === 'application/pdf') {
            const data = await pdfParse(req.file.buffer);
            extractedText = data.text;
        } else if (mimeType.startsWith('image/')) {
            const { data: { text } } = await Tesseract.recognize(req.file.buffer, 'eng');
            extractedText = text;
        } else {
            return res.status(400).json({ error: 'Unsupported file type' });
        }

        res.json({ extractedText });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Error processing file' });
    }
});

app.post('/api/generate-questions', async (req, res) => {
    try {
        const { extractedText, prompt, questionType, questionCount } = req.body;

        if (!extractedText || !questionType || !questionCount) {
            return res.status(400).json({ error: 'Missing parameters' });
        }

        const fullPrompt = `
Generate exactly ${questionCount} questions of type ${questionType} based on the following content:

Content:
"""
${extractedText}
"""

Instructions:
- Generate ${questionCount} questions.
- Question type: ${questionType} (MCQ, Short, Broad)
- For MCQ, provide 4 distinct options (A, B, C, D) and specify the correct answer letter (e.g., "A").
- Ensure all generated questions are directly answerable from the provided content.
- Format your response strictly as a raw JSON array of objects without any markdown or triple backticks.
- Each object in the array should have keys: "question", "options" (array, empty for Short/Broad), and "answer" (string or letter).

Now, generate the questions:
`;

        const generationConfig = {
            temperature: 0.7,
            maxOutputTokens: 2000,
        };

        const result = await model.generateContent({
            contents: [{ role: 'user', parts: [{ text: fullPrompt }] }],
            generationConfig,
        });

        let questionsText = result.response.text();

        let questions = [];

        try {
            const match = questionsText.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
            const cleanedText = match ? match[1] : questionsText;

            const parsed = JSON.parse(cleanedText);

            questions = parsed.map(q => ({
                question: q.question,
                options: q.options || [],
                answer: q.answer || '',
                type: questionType,
            }));
        } catch (parseError) {
            console.warn('❌ Failed to parse JSON from Gemini response, treating as plain text:', questionsText);
            questions = [{
                question: "Could not parse structured response. Here's the raw output:\n" + questionsText,
                options: [],
                type: questionType,
            }];
        }

        res.json({ questions });

    } catch (err) {
        console.error('Error generating questions with Gemini:', err);
        if (err.response && err.response.status === 429) {
            res.status(429).json({ error: 'Rate limit exceeded for Gemini API. Please try again later.' });
        } else {
            res.status(500).json({ error: 'Error generating questions' });
        }
    }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});