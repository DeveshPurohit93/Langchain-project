import argparse
from ingest import ingest_dir
from retriever import SimpleRetriever
from mock_llm import MockLLM

def cmd_ingest(args):
    print(ingest_dir(data_dir=args.dir, store_dir=args.store))

def cmd_ask(args):
    retriever = SimpleRetriever(store_dir=args.store)
    retrieved = retriever.retrieve(args.question, k=args.k)
    llm = MockLLM()
    ans = llm.generate(args.question, retrieved)
    print('\n' + ans)

def main():
    p = argparse.ArgumentParser(description='Simple LangChain tutorial (mini RAG)')
    sub = p.add_subparsers(dest='cmd')

    pi = sub.add_parser('ingest', help='Ingest text files from data/')
    pi.add_argument('--dir', default='data', help='folder with .txt files')
    pi.add_argument('--store', default='store', help='where to save vector store')

    pa = sub.add_parser('ask', help='Ask a question (requires prior ingest)')
    pa.add_argument('question', help='Question text')
    pa.add_argument('--k', type=int, default=3, help='how many chunks to retrieve')
    pa.add_argument('--store', default='store', help='where vector store is saved')

    args = p.parse_args()
    if args.cmd == 'ingest':
        cmd_ingest(args)
    elif args.cmd == 'ask':
        cmd_ask(args)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
