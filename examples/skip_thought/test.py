import unittest
import os, shutil
import main, tools, train
import numpy as np

class TestMain(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		os.mkdir('test_dir')
		os.mkdir('test_dir/1')
		os.mkdir('test_dir/2')
		with open('test_dir/1/1.txt', 'w', encoding = 'utf-8') as f:
			f.write("<doc id=\"7\" url=\"https://ru.wikipedia.org/wiki?curid=7\" title=\"Литва\">\nПроверка раз... Проверка два. Проверка три?! Проверка четыре?")
		
		with open('test_dir/2/1.txt', 'w', encoding = 'utf-8') as f:
			f.write("Литва")
		
		with open('test_dir/2/2.txt', 'w', encoding = 'utf-8') as f:
			f.write("Крупнейшие реки — Неман () и Вилия (). Климат переходный от морского к континентальному.")
		
		with open('test_dir/train.csv', 'w', encoding = 'utf-8') as f:
			lines = ['context_id\tword\tgold_sense_id\tpredict_sense_id\tpositions\tcontext',
					'1\tбазар\t1\t\t28-33\tБыло шесть часов . А мама с базара все не возвращалась . Мы знали что это значит',
					'2\tальбом\t2\t\t275-280\tРомантизм в России дает возможность говорить не столько о русском романтизме сколько о том как его видит советское искусствознание и незаменимым источником является каталог . Пухлые каталоги для сегодняшних выставок норма но тут перед нами не сборник содержательных статей а альбом . Имелось в виду очевидно воспроизвести альбом XIX века',
					'3\tальбом\t1\t\t74-80\tТвой папа Лиза знал меня еще маленькой девочкой . Пойди Лиза достань твои альбомы . Марья Петровна и Акантов остались вдвоем']
			f.write('\n'.join(lines))
			
	
	@classmethod
	def tearDownClass(self):
		shutil.rmtree('test_dir')
	
	def testCorpusContent(self):
		lines = list(main.load_corpus('test_dir'))
		self.assertEqual(lines, ["Проверка раз", "Проверка два", "Проверка три", "Проверка четыре", "Крупнейшие реки Неман Вилия", "Климат переходный от морского континентальному"])
	
	def testTrain(self):
		global model
		global embed_map
		main.save_vocab('test_dir', 'test_dir/dict.dat')
		self.assertTrue(os.path.exists('test_dir/dict.dat'))
		
		embed_map = tools.load_googlenews_vectors('word2vec.w2v', binary = True)
		train.trainer(list(main.load_corpus('test_dir')), saveto = 'test_dir/model', saveFreq = 10, n_words = 10) #you may want to change parameters saveFreq or n_words if you use other test corpus texts
		os.rename('test_dir/model.pkl', 'test_dir/model.npz.pkl')
		self.assertTrue(os.path.exists('test_dir/model.npz'))
		self.assertTrue(os.path.exists('test_dir/model.npz.pkl'))
		
		model = tools.load_model('test_dir/model.npz', 'test_dir/dict.dat', 'word2vec.w2v', embed_map)
		X_train, y_train = main.training_set(model, ['test_dir/train.csv'])
		
		self.assertEqual(len(X_train.shape), 2)
		self.assertEqual(len(y_train.shape), 1)
		self.assertEqual(X_train.shape[0], y_train.shape[0])
		self.assertEqual(X_train.shape[1], 4800)
		
	
	def testMetric(self):
		X_train = np.random.random((100,))
		y_train = np.random.random((100,))
		with self.assertRaises(Exception):
			main.create_metric(X_train, y_train)
		
		X_train = np.random.random((200,1000))
		y_train = np.random.randint(0, 1, (100))
		with self.assertRaises(Exception):
			main.create_metric(X_train, y_train)
		
		X_train = np.random.random((100,))
		y_train = np.random.randint(0, 2, (100,200))
		with self.assertRaises(Exception):
			main.create_metric(X_train, y_train)
		
		X_train = np.random.random((100,1000))
		y_train = np.random.random((100,))
		with self.assertRaises(Exception):
			main.create_metric(X_train, y_train)
		
		X_train = np.random.random((100,1000))
		y_train = np.random.randint(0, 2, (100,))
		metric = main.create_metric(X_train, y_train)
		
		result = metric(np.random.random(500), np.random.random(500))
		self.assertIsInstance(result, (float, np.float32, np.float64))
		self.assertGreaterEqual(result, 0)

	def testTestSet(self):
		os.mkdir('unit_tests')
		try:
			with open('unit_tests/test.csv', 'w', encoding = 'utf-8') as f:
				f.write('1\t2\t3\t4\t5\t6\n1\t2\t3\t4\t5\t6\t7\n1\t2\t3')
			
			metric = main.create_metric(np.random.random((100, 4800)), np.random.randint(0, 2, (100,)))
			with self.assertRaises(Exception):
				main.test_set('unit_tests/test.csv', model, metric, 'unit_test/output.csv')
		finally:
			os.remove('unit_tests/test.csv')
			os.rmdir('unit_tests')

if __name__ == '__main__':
	unittest.main()
