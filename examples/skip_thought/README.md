This folder contains an approach to sense disambiguation task based on the following work: 
Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. "Skip-Thought Vectors."

Files in this folder, except for main.py and test.py, are based on this repository: https://github.com/ryankiros/skip-thoughts, 
however we made some changes, such as adopting the code to Python 3. 

Run 

    python3 main.py

to use this code. You may use command line arguments to specify how the program should work. 
Run 

    python3 main.py -h

or

    python3 main.py --help

to learn how to use them. 

The file test.py contains unit tests. You need a word2vec file 'word2vec.w2v' in the same directory to run the tests.

This program was developed by Vadim Fomin as a part of a group working on Dialogue Evaluation 2018 at the Novosibirsk University.
Feel free to contact me on wadimiusz@gmail.com.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
