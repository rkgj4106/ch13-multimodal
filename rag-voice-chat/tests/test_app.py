import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
import app

class RagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch.object(app, 'DB_PATH', Path(self.tmp.name) / 'test.sqlite3')
        self.patch.start()
        self.client = TestClient(app.app)
        self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.patch.stop()
        self.tmp.cleanup()
    def upload(self):
        with patch.object(app, 'embed', return_value=np.array([[1., 0.]])):
            return self.client.post('/api/documents', files={'file': ('test.txt', 'Pod 로그는 kubectl logs로 확인합니다.'.encode())})
    def test_upload_retrieve_delete(self):
        uploaded = self.upload()
        self.assertEqual(uploaded.status_code, 200)
        with patch.object(app, 'embed', return_value=np.array([[1., 0.]])), patch.object(app, 'ollama_post', return_value={'message': {'content': 'kubectl logs [1]'}}) as call:
            response = self.client.post('/api/chat', json={'question': '로그 확인 방법'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['sources'][0]['name'], 'test.txt')
            self.assertIn('kubectl logs', call.call_args.args[1]['messages'][1]['content'])
        self.client.delete(f'/api/documents/{uploaded.json()["id"]}')
        self.assertEqual(self.client.get('/api/documents').json()['documents'], [])
        with app.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0], 0)
    def test_no_evidence(self):
        with patch.object(app, 'ollama_post') as call:
            response = self.client.post('/api/chat', json={'question': '질문'})
            self.assertIn('근거를 찾지 못했습니다', response.json()['answer'])
            call.assert_not_called()
        self.upload()
        with patch.object(app, 'embed', return_value=np.array([[0., 1.]])), patch.object(app, 'ollama_post') as call:
            self.assertEqual(self.client.post('/api/chat', json={'question': '다른 주제'}).json()['sources'], [])
            call.assert_not_called()
    def test_embedding_failure_atomic(self):
        with patch.object(app, 'embed', side_effect=app.HTTPException(502, '실패')):
            self.assertEqual(self.client.post('/api/documents', files={'file': ('a.txt', b'text')}).status_code, 502)
        self.assertEqual(self.client.get('/api/documents').json()['documents'], [])
    def test_duplicate_and_model_isolation(self):
        self.upload(); self.upload()
        self.assertEqual(len(self.client.get('/api/documents').json()['documents']), 1)
        with patch.object(app, 'EMBED_MODEL', 'different'), patch.object(app, 'embed') as call:
            self.assertEqual(app.retrieve('질문', 4), [])
            call.assert_not_called()
    def test_validation_and_general_chat(self):
        self.assertEqual(self.client.post('/api/documents', files={'file': ('x.exe', b'text')}).status_code, 400)
        self.assertEqual(self.client.post('/api/chat', json={'question': '  '}).status_code, 400)
        self.assertEqual(self.client.post('/api/chat', json={'question': '질문', 'top_k': 9}).status_code, 422)
        with patch.object(app, 'ollama_post', return_value={'message': {'content': '일반 답변'}}):
            self.assertEqual(self.client.post('/api/chat', json={'question': '질문', 'use_rag': False}).json()['answer'], '일반 답변')
    def test_invalid_embedding(self):
        for value in ([], [[0., 0.]], [[float('nan'), 1.]], [[1.], [2.]]):
            with patch.object(app, 'ollama_post', return_value={'embeddings': value}):
                with self.assertRaises(app.HTTPException): app.embed(['text'])
        with patch.object(app, 'ollama_post', return_value={'embeddings': [[3., 4.]]}):
            np.testing.assert_allclose(app.embed(['text']), [[.6, .8]], atol=1e-6)

if __name__ == '__main__': unittest.main()
