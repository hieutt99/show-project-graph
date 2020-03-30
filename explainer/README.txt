I edited a little bit in some files in order to run my code

DATALOADER.py --> def "get_page":
link = "../data/task1train_pro/"+name+".txt"
label_link = "../data/task2train_pro/"+name+".json"
with open('../bow/docs.json','r') as fp:
(all added a stop at the begining of the url) ( "." ---> "..")


HOW TO RUN???
1. The test will be taken in "data/task1train_pro"
2. In "main.py",
    + Line 24: Replace "1" with the number of the test you want to test. (index from 0, 1, 2, ...)
    + Line 25: Replace the ID at the end with the corresponding ID of the test. (Eg: test 0 - X00016469612, test 1 - X00016469619)
3. Run "main.py" and wait :))))

Note that only 1 test will be tested per time.

