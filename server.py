
from __future__ import unicode_literals
import json
import requests
from flask import Flask
from flask.ext.restful import Resource, Api, reqparse
import argparse
import base64
from gensim.models.word2vec import Word2Vec, Vocab
import numpy as np
import sys

'''
Example call: curl http://127.0.0.1:5000/catsim/similar/catid=123&n=5
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
'''

parser = reqparse.RequestParser()

service = "http://0.0.0.0:8001/category/get_category_space?"

class CategorySimilarity:

    def __init__(self):
        self.category_space = self.build_category_space()

    def is_valid_vector(self, vector):
        vstr = ''
        wrong_value = False
        if vector.any() and vector.size == 300:
            for val in vector:
                #print "%.32f " % val
                vstr += "%.32f " % val
                if (np.nan_to_num(val) == 0) or (val < -1.5) or (val > 1.5) or (val > -0.000000000001 and val < 0.000000000001):
                    wrong_value = True
                    break
            if not wrong_value:
                return vstr
        return False

    # def save_category_space_to_disk(self, cs_string):
    #     space = open(settings.CATEGORY_SPACE, 'w')
    #     space.write(cs_string)
    #     space.close()

    def add_vector_to_model(self, category_id, vector, model):
        # The category should not already be in the space
        # (rebuild the space in that case)
        catid = '#' + unicode(category_id)
        if catid in model.vocab:
            self.remove_category_from_space(category_id)
        w_count = len(model.vocab)
        model.vocab[catid] = Vocab(index=w_count, count=w_count+1)
        model.index2word.append(catid)
        if w_count == 0:
            model.syn0 = np.empty((1, 300), dtype=np.float32)
            model.syn0[0] = vector
        else:
            try:
                model.syn0 = np.vstack((model.syn0, vector))
            except ValueError as e:
                print(e)
                print("Vector length: {}".format(len(vector)))
                print("Space Length: {}".format(model.vector_size))
        return model

    def build_category_space(self):
        """
        @summary retrieve all category vectors and construct a Word2Vec vector space.
        # Compute the vector if it wasn't computed.
        """
        category_space = Word2Vec(size=300)
        s = requests.Session()
        s.headers = {"content-type": "application/json", "accept": "application/json"}
        resp = s.get(service + "page=1")
        data = json.loads(resp.content)
        num_pages = data['num_pages']
        categories = data['categories_with_vectors']
        page = 1
        while page <= num_pages:
            for category in categories:
                # Get the vector and add it to the space
                if category['_vector']:
                    vector = np.fromstring(base64.b64decode(category['_vector']))
                    if not vector.any():
                        print "invalid vector for category {}".format(category['id'])
                        continue
                    v = self.is_valid_vector(vector)
                    if v:
                        # print category['id']
                        category_space = self.add_vector_to_model(category['id'], vector, category_space)
                    else:
                        print "invalid vector for category {}".format(category['id'])
                        continue
            page += 1
            resp = s.get(service + "page={}".format(page))
            data = json.loads(resp.content)
            categories = data['categories_with_vectors']
        print("Space Length: {}".format(str(len(category_space.vocab.keys()))))
        # print category_space.vocab
        return category_space

    def add_category_to_space(self, category_id, vector, space):
        """
        Updates the category space with a new category, update the cache.
        """
        vector = np.fromstring(vector)
        if not vector.any():
            print "invalid vector for category {}".format(category_id)
            return
        v = self.is_valid_vector(vector)
        if v:
            print category_id
            print vector
            print space.vector_size
            space = self.add_vector_to_model(category_id, vector, space)
            self.category_space = space
            print("new category added to vector space.")
        else:
            print "invalid vector for category {}".format(category_id)
            return
        return space

    def remove_category_from_space(self, category_id):
        """
        :param category: The category to be removed
        :return: None
        Also updates cache
        """
        try:
            voc = self.category_space.vocab['#{}'.format(category_id)]
        except KeyError:
            print("Could not remove category {} from the space. It was not there.".format(category_id))
        else:
            del self.category_space.vocab['#{}'.format(category_id)]
            self.category_space.syn0 = np.delete(self.category_space.syn0, voc.index)

    def most_similar_categories(self, category_ids, n=3):
        """
        @summary: returns the n most similar categories to the category given.
        """
        try:
            if type(category_ids) == list:
                tops = self.category_space.most_similar_cosmul(positive=['#' + unicode(category_id) for category_id in category_ids], topn=n)
            else:
                tops = self.category_space.most_similar_cosmul('#' + unicode(category_ids), topn=n)
            return [top[0].lstrip('#') for top in tops]
        except AttributeError as e:
            print(unicode(e) + ": " + unicode(category_ids))
            return []
        except TypeError as e:
            print(str(e))
        except KeyError as e:
            print(str(e))
        except:
            e = sys.exc_info()[0]
            print(str(e))
        print "Category {}".format(category_ids)
        return []

cs = CategorySimilarity()

class Add(Resource):
    """
    Adds a category vector to the model
    """
    def post(self):
        parser.add_argument('vector', type=str, required=True, help="base64 encoding of the Numpy vector string export for the category")
        parser.add_argument('catid', type=str, required=True, help="Category ID is mandatory")
        args = parser.parse_args()
        print args
        vector = base64.b64decode(args['vector'])
        space = cs.add_category_to_space(args['catid'], vector, cs.category_space)
        if space:
            cs.category_space = space
        return


class Del(Resource):
    """
    Remove a category
    """
    def post(self):
        parser.add_argument('catid', type=str, required=True, help="Category ID is mandatory")
        args = parser.parse_args()
        cs.remove_category_from_space(args['catid'])
        # print """           +++++++++++++++++++++++++++++++++++++
        #    +                                   +
        #    +            YOUPI                  +
        #    +                                   +
        #    +++++++++++++++++++++++++++++++++++++"""
        return


class Similar(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('catid', type=int, required=True, help="Category ID is mandatory", action='append')
        parser.add_argument('n', type=int, required=False, help="Number of similar categories to return")
        args = parser.parse_args()
        catid = args['catid']
        if args['n']:
            n = args['n']
        else:
            n = 3
        similar = cs.most_similar_categories(catid, n)
        print "{}".format(similar)
        return similar

app = Flask(__name__)
api = Api(app)

@app.errorhandler(404)
def pageNotFound(error):
    return "page not found"

@app.errorhandler(500)
def raiseError(error):
    return error

if __name__ == '__main__':

    # ----------- Parsing Arguments ---------------
    p = argparse.ArgumentParser()
    p.add_argument("--host", help="Host name (default: localhost)")
    p.add_argument("--port", help="Port (default: 5000)")
    p.add_argument("--path", help="Path (default: /catsim)")
    args = p.parse_args()
    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/catsim"
    port = int(args.port) if args.port else 5000
    api.add_resource(Similar, path + '/similar')
    api.add_resource(Add, path + '/add')
    api.add_resource(Del, path + '/del')

    app.run(host=host, port=port)

