"""
@author : MaJiaHao
@when : 2025-4-13
@homepage : https://github.com/HEX-QWQ/LucaOne
"""
from torch import nn

from models.embedding.similarity_embedding import SimilarityEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + similarity encoding (sinusoid)
    similarity encoding can give similarity information to network
    """

    def __init__(self, vocab_size, d_model, drop_prob, device):
        """
        class for word embedding that included similarity information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.sim_emb = SimilarityEncoding(d_model, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.sim_emb(x)
        return self.drop_out(tok_emb + pos_emb)
