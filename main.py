from datapreprocessor import preprocessor
from forests.CART import CART

if __name__ == '__main__':
    # CART Test
    cart = CART()
    cart.fit(preprocessor.load_test_dataset())

