from pprint import pprint as pp

from sauveur.sauveur import RAG
# from sauveur.chunking import Chunking

obj1 = RAG()

# a = ['oiwnfeorw','wefwef']
# a = [['wedw','wefgr','rtbnrt','erbef'],
#     ['nrt','edfb','asfc','bswg'],
#     ['owie','wedwf','debgg','aerg'],

# ]
# x= obj1.create_chunks(items=[(12245,3543), (1579,672)], chunk_size=2)
x= obj1.create_chunks(items=['aser3543','1erg9672'], chunk_size=2)
# x = obj1.create_chunks(items=[[None, True, 64864]], chunk_size=2)
# x = obj1.create_chunks(items=[{'w':('None', 'True', '64864'),'r':(None, False, 46865)}], chunk_size=2)
# print(type(x),slen(x[0]['chunks']))#,x)#[0])
pp(x)
# y = x[0]['chunks'][:5]
# pp(y)
# print(y[0]['chunks'])
e=x[0]['chunks'][:2]
print([obj1.generate_embeddings(
#     items=y[0]['chunks'],
    docs=e[0]#[13513],
    # model_provider='google',
    # model_name='text-embedding-004',#gemini-embedding-001',
    # model_dimension=768,
    # model_provider_kwargs={
        # 'encode_kwargs':{"normalize_embeddings": True}
    #     'google_api_key':'AIzaSyB1Y8nopTd5kQYEvah7HlKbUFR2t5U-tns',#AIzaSyAWJyrQTIxhr-GjGP-5v2n8djs3FQqRrjo'
    # }
)[0][:5]],'...')
# obj2 = Chunking()
# y = obj2.x([[[['wefwsefsd','fubwibgw','wekbwbr3r'],'mphojkm']]],3)

# print(y)

# print(obj1.create_bulk_object('ef',['','']))
# docs = []
# for i in range(10):
#     doc = {
#             'action': 'create',
#             'index': 'index1',
#             'doc_id': f'id{i}',
#             # 'data': {f'data{i}': f"{'-'*20}data{i}{'-'*20}"},
#             'data': {
#                 'x':{
#                     'y': 25887
#                 }
#             }
#         }
#     docs += [doc]
# pp(docs)
# x = obj1.create_bulk_objects(docs=docs, no_of_docs_per_bulk_object=4)
# print(x,len(x))

# from opensearchpy import OpenSearch
# pp(obj1.similaity_search(
#     opensearch_client=OpenSearch(),
#     index='idx1',
#     embeddings_field_name='emb',
#     query_embeddings=[22313,351],
#     k=10,Top=10,
#     query_object_attributes={'match':{'gt':None}},
#     source_object_attributes={
#         "_source": {
#             "includes": ["page_content"]
#     }}
# ))

# pp(obj1.combine_chunked_docs([
#     {'chunks': ['wefwsefsd','fubwibgw','wekbwbr3r']},
#     # {'chunks': ['wefwsefsd','fubwibgw','wekbwbr3r']},
# ],[]))



