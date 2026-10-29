import tempfile
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import app

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = patch.object(app,'DATA',Path(self.tmp.name))
        self.p.start()
        app.init_db()
        self.client = TestClient(app.app)
    def tearDown(self):
        self.p.stop(); self.tmp.cleanup()
    def test_enqueue_and_result(self):
        with patch.object(app.executor,'submit') as submit:
            response=self.client.post('/api/generate',json={'prompt':'robot','translate':False})
        app.slots.release()
        self.assertEqual(response.status_code,202)
        job_id=response.json()['job_id']
        submit.assert_called_once()
        self.assertEqual(self.client.get('/api/jobs/'+job_id).json()['status'],'queued')
        self.assertEqual(self.client.get('/api/jobs/'+job_id+'/image').status_code,404)
        app.update(job_id,'done',{'seed':1})
        (app.DATA/f'{job_id}.png').write_bytes(b'test png bytes')
        result=self.client.get('/api/jobs/'+job_id+'/image')
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.headers['content-type'],'image/png')
    def test_validation(self):
        self.assertEqual(self.client.post('/api/generate',json={'prompt':'   '}).status_code,400)
        self.assertEqual(self.client.post('/api/generate',json={'prompt':'robot','width':513}).status_code,400)
        self.assertEqual(self.client.post('/api/generate',json={'prompt':'robot','steps':100}).status_code,422)
        self.assertEqual(self.client.get('/api/jobs/missing').status_code,404)
    def test_restart_marks_failed(self):
        with app.db() as conn:
            conn.execute('INSERT INTO jobs(id,status,request) VALUES(?,?,?)',('test','running','{}'))
        app.init_db()
        self.assertEqual(app.get_job('test')['status'],'failed')
    def test_worker_success(self):
        with app.db() as conn:
            conn.execute('INSERT INTO jobs(id,status,request) VALUES(?,?,?)',('success','queued','{}'))
        app.slots.acquire()
        image=Mock()
        image.save.side_effect=lambda path,format: Path(path).write_bytes(b'generated png placeholder')
        tokenizer=Mock(return_value={'input_ids':[1,2]})
        tokenizer.model_max_length=77
        pipeline=Mock(return_value=SimpleNamespace(images=[image]))
        pipeline.tokenizer=tokenizer
        pipeline.tokenizer_2=tokenizer
        generator=Mock()
        generator.manual_seed.return_value=generator
        torch=SimpleNamespace(Generator=Mock(return_value=generator),inference_mode=nullcontext)
        with patch.object(app,'load_pipeline',return_value=pipeline), patch.dict(sys.modules,{'torch':torch}):
            app.run_job('success',app.GenerateRequest(prompt='robot',translate=False,seed=42))
        result=app.get_job('success')
        self.assertEqual(result['status'],'done')
        self.assertEqual(result['result']['seed'],42)
        self.assertEqual(pipeline.call_args.kwargs['prompt'],'robot')
        self.assertEqual(pipeline.call_args.kwargs['width'],1024)
        self.assertTrue((app.DATA/'success.png').exists())

    def test_translation_failure(self):
        with app.db() as conn:
            conn.execute('INSERT INTO jobs(id,status,request) VALUES(?,?,?)',('test','queued','{}'))
        app.slots.acquire()
        with patch.object(app,'translate_prompt',side_effect=RuntimeError('Ollama가 빈 프롬프트를 반환했습니다.')):
            app.run_job('test',app.GenerateRequest(prompt='로봇',translate=True))
        self.assertEqual(app.get_job('test')['status'],'failed')

if __name__=='__main__': unittest.main()
