# Conversa com o Claude Code (uso de IA no TP2)

Transcrição da sessão do Claude Code (modelo Claude Opus 5.5) usada durante o TP2 de Introdução à Inteligência Artificial (DCC642, UFMG).

- Exportada automaticamente do log local da sessão; o raciocínio interno do modelo não é incluído.
- Chamadas de ferramentas (comandos executados, arquivos lidos/editados) aparecem resumidas; saídas longas foram truncadas.
- Mensagens do usuário estão sem edição.



---

## 👤 Aluno

Estou implementando o passo a passo como descrito no pdf em docs. Avalie minha funcao initialize_parameters para ver se ficou correta e me ajude a implementar a RELU. Eu sei como a relu funciona, porém nao sei manipular arrays do numpy o suficiente para tal. Em pseudocodigo eu faria algo como: para cada i, j de Z, eu rodaria um max Z[i][j], 0 e salvaria o resultado na matriz, devolvendo ela.


---

## 🤖 Claude

**Ferramenta: Bash** — List project files and docs

```bash
ls -la && ls docs && find . -name "*.py" -not -path "./.venv/*" -not -path "*/site-packages/*" | head -20
```

<details><summary>resultado</summary>

```
total 20
drwxr-xr-x 5 vinicius vinicius 4096 Nov 17 16:46 .
drwxr-xr-x 4 vinicius vinicius 4096 Nov 17 13:20 ..
drwxr-xr-x 2 vinicius vinicius 4096 Nov 17 13:21 docs
drwxr-xr-x 6 vinicius vinicius 4096 Nov 17 16:47 src
drwxr-xr-x 6 vinicius vinicius 4096 Nov 17 16:46 tp2-env
tp2-rna.pdf
./src/plot.py
./src/server.py
./src/learn.py
```

</details>

**Ferramenta: Bash** — Show learn.py contents

```bash
cat -n src/learn.py && ls src
```

**Ferramenta: Read** — `/home/vinicius/ufmg/tp-2-intro-ia/docs/tp2-rna.pdf`

<details><summary>resultado</summary>

```
PDF file read: /home/vinicius/ufmg/tp-2-intro-ia/docs/tp2-rna.pdf (822.9KB)
[documento PDF]
```

</details>

<details><summary>resultado</summary>

```
1	import numpy as np
     2	np.random.seed(42)
     3	
     4	from sklearn.datasets import fetch_openml
     5	from sklearn.model_selection import train_test_split
     6	from sklearn.preprocessing import OneHotEncoder
     7	
     8	def load_mnist():
     9	    # Load MNIST data from sklearn
    10	    mnist = fetch_openml('mnist_784', as_frame=False, cache=True, version=1)
    11	    X, y = mnist["data"], mnist["target"].astype(int)
    12	
    13	    # Normalize the data
    14	    X = X / 255.0
    15	
    16	    # Split the data into training and testing sets
    17	    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    18	
    19	    # One-hot encode the labels
    20	    encoder = OneHotEncoder(sparse=False)
    21	    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    22	    y_test = encoder.transform(y_test.reshape(-1, 1))
    23	
    24	    return X_train, X_test, y_train, y_test
    25	
    26	def initialize_parameters(input_size, hidden_size, output_size):
    27	    W1 = np.random.randn(input_size, hidden_size)
    28	    b1 = np.zeros(1, hidden_size)
    29	    W2 = np.random.randn(hidden_size, output_size)
    30	    b2 = np.zeros(1, output_size)
    31	    return (W1, b1, W2, b2)
    32	
    33	# Activation functions
    34	def relu(Z):
    35	    pass
    36	
    37	def relu_derivative(Z):
    38	    return Z > 0
    39	
    40	def softmax(Z):
… (54 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Check np.zeros call and seed output

```bash
tp2-env/bin/python -c "
import numpy as np
try:
    np.zeros(1, 4)
except Exception as e:
    print('np.zeros(1, 4) ->', type(e).__name__, e)
np.random.seed(42)
W1 = np.random.randn(784, 4)
print('primeira linha sem *0.01:', W1[0])
"
```

<details><summary>resultado</summary>

```
np.zeros(1, 4) -> TypeError Cannot interpret '4' as a data type
primeira linha sem *0.01: [ 0.49671415 -0.1382643   0.64768854  1.52302986]
```

</details>

## `initialize_parameters`: tem dois problemas

```python
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros(1, hidden_size)
```

1. **Falta o `* 0.01` em `W1` e `W2`.** O PDF (seção 3.1) pede pesos pequenos, perto de zero. Sem esse fator, a primeira linha de `W1` sai `[0.4967, -0.1383, 0.6477, 1.5230]`, e o esperado é `[0.00496714, -0.00138264, ...]`. Fora isso a seed está certa: os números batem com o esperado depois de multiplicar por 0.01.

2. **`np.zeros(1, hidden_size)` gera erro.** O `np.zeros` recebe o shape como **um único argumento** (uma tupla). O segundo argumento posicional é o `dtype`. Testei no seu venv e deu:
   ```
   TypeError: Cannot interpret '4' as a data type
   ```
   O NumPy entendeu o `4` como tipo de dado. O certo é passar a tupla: `np.zeros((1, hidden_size))`, com parênteses duplos. Vale o mesmo para o `b2`.

O `randn` aceita as dimensões soltas (`randn(784, 4)`), mas o `zeros` exige a tupla. Essa diferença de API costuma confundir.

## ReLU: do seu pseudocódigo para NumPy

O seu raciocínio funciona e daria para escrever com dois `for` sobre `Z.shape`. No NumPy, porém, quase sempre dá para trocar o loop por uma **operação vetorizada**, que age sobre o array inteiro de uma vez e roda muito mais rápido. Para MNIST com dezenas de milhares de linhas e milhares de épocas, essa diferença conta.

Há duas ferramentas que cabem aqui:

**1. `np.maximum(a, b)`**: faz o máximo **elemento a elemento** entre dois arrays. O ponto principal é que, se um dos lados for um escalar, o NumPy faz *broadcasting*: compara cada elemento com aquele escalar.
```python
>>> x = np.array([[3, 7], [9, 1]])
>>> np.maximum(x, 5)
array([[5, 7],
       [9, 5]])
```
Não confunda com `np.max`: ele devolve **um único número**, o maior do array inteiro (ou de um eixo). O que você quer é o `np.maximum`.

**2. Máscara booleana**: você já tem uma no arquivo. O `relu_derivative` faz `Z > 0`, que devolve um array de `True`/`False` com o mesmo shape de `Z`. Multiplicar por ela zera onde a máscara é `False`:
```python
>>> x = np.array([[3, 7], [9, 1]])
>>> x * (x < 5)
array([[3, 0],
       [0, 1]])
```

Com uma dessas duas ideias a ReLU sai em uma linha. Deixo a escrita para você por causa da seção 6 do PDF: ela proíbe que a IA gere a implementação das funções do `learn.py`. Ajuda para entender e conferir é permitida, desde que você declare no relatório.

Para conferir, o esperado é:
```python
learn.relu(np.array([[-1, 0, 1], [-2, 0, 2]]))
# [[0 0 1]
#  [0 0 2]]
```
Use um array de inteiros, como no exemplo, e a saída deve continuar inteira. Se aparecer `0.` ou `True`, a operação mudou o tipo do array.

Depois de escrever, se quiser, eu rodo os "Resultados esperados" do PDF contra o seu `learn.py` para validar a inicialização e a ReLU.


---

## 👤 Aluno

O que o Z representa aqui?


---

## 🤖 Claude

**Skill: teach** — O que o Z representa no learn.py (relu(Z), forward Z1 = XW1 + b1, Z2 = A1W2 + b2) da RNA de duas camadas do TP2

<details><summary>resultado</summary>

```
Launching skill: teach
```

</details>

_(instruções da skill `teach` carregadas)_

O `Z` é a **pré-ativação** de uma camada. Ele guarda o valor "bruto" de cada neurônio, a soma ponderada das entradas mais o bias, **antes** de passar pela função de ativação. O `A` é esse mesmo valor **depois** da ativação.

```
Z1 = X·W1 + b1     →  A1 = relu(Z1)       (camada escondida)
Z2 = A1·W2 + b2    →  A2 = softmax(Z2)    (camada de saída)
```

**Analogia com o Nest.** Pense no `Z` como o payload que chega no controller e no `A` como o que sai de um `Pipe` (tipo `ParseIntPipe`). Primeiro vem o dado cru, depois uma transformação fixa aplicada em cima dele. A analogia para por aí. A ativação não valida nada. Ela existe para introduzir **não linearidade**. Sem a ReLU, `Z2 = (X·W1)·W2` seria só outra multiplicação de matrizes, e as duas camadas virariam uma única camada linear. A rede não conseguiria aprender fronteiras curvas entre os dígitos.

**Como funciona aqui, com os números do PDF.**

- `X` tem shape `(m, 784)`: cada **linha** é uma imagem achatada.
- `W1` tem shape `(784, hidden)` e `b1` tem shape `(1, hidden)`.
- Então `Z1` tem shape `(m, hidden)`. A linha `i` é a imagem `i` e a coluna `j` é o neurônio `j`.
- Cada célula é `Z1[i][j] = Σₖ X[i][k]·W1[k][j] + b1[j]`. Em palavras: o quanto o neurônio `j` "acende" para a imagem `i`.

No exemplo da seção 3.3 (2 imagens, 4 neurônios):

```
Z1 = [[ 0.2424, -0.1806, -0.3202,  0.0505],
      [-0.3854, -0.1333,  0.0295,  0.1388]]

A1 = [[ 0.2424,  0.    ,  0.    ,  0.0505],    ← relu zerou os negativos
      [ 0.    ,  0.    ,  0.0295,  0.1388]]
```

Na saída, `Z2` tem shape `(m, 10)`, um valor por dígito. Esses valores também têm nome próprio: **logits**, pontuações ainda não normalizadas. A softmax transforma cada linha em probabilidades que somam 1. No exemplo, todas ficam perto de `0.1`. Faz sentido, porque com pesos de ordem `0.01` a rede ainda não tem opinião nenhuma e chuta 10% para cada dígito.

**Por que o `forward` devolve o `Z` e não só o `A`.** O seu `backward` precisa dele na linha `src/learn.py:60`:

```python
dZ1 = dA1 * relu_derivative(Z1)
```

A derivada da ReLU depende de o neurônio ter ficado positivo **antes** da ativação. Por isso o `Z1` precisa ser guardado.

**Armadilhas:**
- **ReLU só em `Z1`.** Em `Z2` vai a softmax. Aplicar ReLU na saída é um erro comum.
- **O `Z` de `relu(Z)` é só o nome do parâmetro.** A função não sabe de que camada o array veio. É por isso que o teste do PDF passa um array qualquer, `[[-1, 0, 1], [-2, 0, 2]]`.
- **Broadcasting do bias.** `b1` tem shape `(1, hidden)` e é somado a **todas** as `m` linhas de `X·W1`. Esse é o motivo de o `np.zeros` precisar da tupla `(1, hidden_size)`: com shape errado, a soma quebra ou faz broadcast do jeito errado sem avisar.
- **Notação transposta.** Muitos livros e cursos (Ng, Nielsen) põem exemplos em **colunas** e escrevem `Z = W·X + b`. O TP usa exemplos em **linhas**, `Z = X·W + b`. É a mesma ideia com os shapes invertidos, então cuidado ao copiar fórmulas de outras fontes.

**Para ler depois:** Michael Nielsen, *Neural Networks and Deep Learning*, cap. 2 (http://neuralnetworksanddeeplearning.com/chap2.html). Lá o `z` aparece com o nome de *weighted input*, e o capítulo mostra por que ele é a peça central do backpropagation. Lembre que ele usa a notação em colunas.

Uma verificação opcional: se `hidden_size = 64` e você passar o treino inteiro (56.000 imagens), qual é o shape de `Z1`, e o que significa a célula `Z1[100][7]`?


---

## 👤 Aluno

Ok, agora sobre o softmax, me explique como ela funciona e como posso implementa-la em numpy.


---

## 🤖 Claude

**Ferramenta: Bash** — Demonstrate numpy sum axis/keepdims and exp overflow

```bash
tp2-env/bin/python -c "
import numpy as np
M = np.array([[1., 2., 3.], [10., 20., 30.], [100., 200., 300.]])
print('sum axis=1           ->', np.sum(M, axis=1), np.sum(M, axis=1).shape)
print('sum axis=1 keepdims  ->', np.sum(M, axis=1, keepdims=True).tolist(), np.sum(M, axis=1, keepdims=True).shape)
print('M / sem keepdims:'); print(M / np.sum(M, axis=1))
print('M / com keepdims:'); print(M / np.sum(M, axis=1, keepdims=True))
print(np.exp(np.array([709., 710., 1000.])))
a = np.exp(np.array([1000., 1001.])); print(a / a.sum())
"
```

<details><summary>resultado</summary>

```
<string>:8: RuntimeWarning: overflow encountered in exp
<string>:9: RuntimeWarning: overflow encountered in exp
<string>:9: RuntimeWarning: invalid value encountered in divide
sum axis=1           -> [  6.  60. 600.] (3,)
sum axis=1 keepdims  -> [[6.0], [60.0], [600.0]] (3, 1)
M / sem keepdims:
[[1.66666667e-01 3.33333333e-02 5.00000000e-03]
 [1.66666667e+00 3.33333333e-01 5.00000000e-02]
 [1.66666667e+01 3.33333333e+00 5.00000000e-01]]
M / com keepdims:
[[0.16666667 0.33333333 0.5       ]
 [0.16666667 0.33333333 0.5       ]
 [0.16666667 0.33333333 0.5       ]]
[8.21840746e+307             inf             inf]
[nan nan]
```

</details>

A softmax recebe uma linha de pontuações quaisquer (os logits do `Z2`) e devolve **probabilidades**: valores entre 0 e 1 que somam 1 na linha. Ela converte "o quanto a rede acha que é cada dígito" em "a probabilidade de ser cada dígito".

**Analogia com o Grafana.** É parecido com montar o painel de "% do tráfego por rota", em que cada rota vira `valor / soma(valores)`. A softmax faz a mesma normalização com um passo antes: aplica `eˣ` em cada valor e só depois divide pela soma. A analogia quebra justamente nesse passo. O `eˣ` deixa tudo positivo, então logits negativos funcionam. Ele também **amplifica as diferenças**: quem está um pouco à frente fica com uma fatia bem maior. Por isso o resultado não é proporcional aos logits.

### O mecanismo, com o exemplo do PDF

`softmax([1, 2, 3])`:

| | z=1 | z=2 | z=3 | soma |
|---|---|---|---|---|
| `eᶻ` | 2.718 | 7.389 | 20.086 | 30.193 |
| `eᶻ / soma` | **0.0900** | **0.2447** | **0.6652** | 1 |

Esses são os números esperados. A segunda linha do teste, `[4, 5, 6]`, dá **exatamente o mesmo resultado**, e isso não é coincidência. Ela é a primeira linha somada de 3. Como `e^(z+3) = e^z · e^3`, o fator `e^3` aparece no numerador e no denominador e se cancela. Conclusão: **somar uma constante à linha inteira não muda a softmax.** Essa propriedade vai ser útil mais abaixo.

### As peças do NumPy

A conta tem três passos: exponenciar, somar **por linha** e dividir cada linha pela sua soma.

**1. `np.exp(arr)`** é elemento a elemento, como o `np.maximum` da ReLU.

**2. `np.sum(arr, axis=..., keepdims=True)`**: você já tem esse padrão no `backward`, em `src/learn.py:57`:
```python
db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
```
- `axis=0` soma **descendo pelas linhas**: uma soma por coluna. Ali ela soma sobre os exemplos.
- `axis=1` soma **atravessando as colunas**: uma soma por linha. É o que a softmax precisa, a soma das 10 classes de cada exemplo.

**3. `keepdims=True` é a armadilha principal.** Rodei isso no seu venv:
```python
M = [[  1,   2,   3],
     [ 10,  20,  30],
     [100, 200, 300]]

np.sum(M, axis=1)                 # [6, 60, 600]        shape (3,)
np.sum(M, axis=1, keepdims=True)  # [[6], [60], [600]]  shape (3, 1)
```
Agora a divisão de `M` por cada uma delas:
```
sem keepdims (errado):           com keepdims (certo):
[[ 0.1667  0.0333  0.005 ]       [[0.1667 0.3333 0.5]
 [ 1.6667  0.3333  0.05  ]        [0.1667 0.3333 0.5]
 [16.6667  3.3333  0.5   ]]       [0.1667 0.3333 0.5]]
```
Com shape `(3,)`, o broadcasting divide **cada coluna** por um valor diferente. O resultado está errado e **não aparece erro nenhum**. Com shape `(3, 1)`, cada **linha** é dividida pela sua própria soma. No MNIST, `Z2` tem shape `(m, 10)` e a soma sem `keepdims` teria shape `(m,)`. Na maioria dos casos isso estoura com `ValueError`. Se `m` for 10, o erro passa calado.

### Estabilidade numérica

O `float64` só aguenta `e^709`. Também rodei isso:
```python
np.exp([709, 710, 1000])   # [8.2e+307, inf, inf]  + RuntimeWarning: overflow
a = np.exp([1000, 1001]); a / a.sum()   # [nan, nan]
```
Se os logits crescerem durante o treino (com taxa de aprendizado 0.5, por exemplo), você recebe `inf / inf = nan`, e o `nan` contamina a loss e todos os pesos.

A correção usa a propriedade da tabela. Como somar ou subtrair uma constante da linha não muda o resultado, **subtraia o máximo de cada linha antes do `exp`**. O maior valor vira 0 (`e⁰ = 1`) e os demais ficam negativos, então nunca há overflow. Para isso servem `np.max(arr, axis=1, keepdims=True)` e o mesmo cuidado com o `keepdims`. O teste do PDF passa com ou sem esse passo, mas no treino real ele é a diferença entre convergir e virar `nan`.

### Outras armadilhas
- **`np.max` × `np.maximum`**: aqui você quer o `np.max` com `axis`, que dá o maior valor *por linha*. Na ReLU era o `np.maximum`, que compara elemento a elemento.
- **Por que `dZ2 = A2 - y` no `backward` é tão simples**: as derivadas da softmax e da entropia cruzada, combinadas, se reduzem a essa subtração. Ela só vale se `A2` for de fato a softmax de `Z2`. Uma softmax errada (somando no eixo errado, por exemplo) estraga o gradiente sem gerar erro.

Como na ReLU, deixo a montagem para você por causa da seção 6 do PDF. Com as três peças acima e o passo de estabilidade, a função fica em 2 ou 3 linhas. Para conferir:
```python
learn.softmax(np.array([[1, 2, 3], [4, 5, 6]]))
# [[0.09003057 0.24472847 0.66524096]
#  [0.09003057 0.24472847 0.66524096]]
```
Um teste extra que pega o erro do `keepdims`: `learn.softmax(...).sum(axis=1)` precisa dar `[1., 1.]`.

**Para ler depois:** CS231n de Stanford, seção *Softmax classifier* (https://cs231n.github.io/linear-classify/#softmax). Ela cobre a intuição, o truque de estabilidade com o máximo e a ligação com a entropia cruzada, que é a sua próxima função.

Uma verificação opcional: por que subtrair o **máximo** da linha, e não a média, garante que o `exp` nunca estoura?


---

## 👤 Aluno

implementei a relu e softmax, roda os testes do pdf


---

## 🤖 Claude

**Ferramenta: Bash** — Show current initialize, relu, softmax code

```bash
sed -n 26,50p src/learn.py
```

<details><summary>resultado</summary>

```
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return (W1, b1, W2, b2)

# Activation functions
def relu(Z):
  return np.maximum(Z, 0)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    e_z = np.exp(Z)
    return e_z / np.sum(e_z, axis=1, keepdims=True)

# Forward pass
def forward(X, parameters):
    pass

# Loss function
def cross_entropy_loss(y_true, y_pred):
    pass
```

</details>

**Ferramenta: Bash** — Run PDF expected-result tests for init, relu, softmax

```bash
mkdir -p /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad && cat > /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py <<'EOF'
import sys, warnings
sys.path.insert(0, "/home/vinicius/ufmg/tp-2-intro-ia/src")
import numpy as np
import learn

def check(name, got, exp, atol=1e-8):
    ok = np.allclose(got, exp, atol=atol)
    print(f"[{'OK' if ok else 'FALHOU'}] {name}")
    if not ok:
        print("  obtido:  ", got); print("  esperado:", exp)

# 3.1 initialize_parameters
W1, b1, W2, b2 = learn.initialize_parameters(784, 4, 10)
print("shapes:", W1.shape, b1.shape, W2.shape, b2.shape)
check("3.1 shapes", [W1.shape == (784, 4), b1.shape == (1, 4), W2.shape == (4, 10), b2.shape == (1, 10)], [True]*4)
check("3.1 W1 primeiras linhas", W1[:3], [[0.00496714, -0.00138264, 0.00647689, 0.0152303],
                                         [-0.00234153, -0.00234137, 0.01579213, 0.00767435],
                                         [-0.00469474, 0.0054256, -0.00463418, -0.0046573]])
check("3.1 W1 últimas linhas", W1[-3:], [[-0.00139397, 0.00739653, -0.01909356, 0.01318302],
                                        [0.00072681, -0.00411493, -0.00089234, -0.00037571],
                                        [-0.01731201, 0.01494938, 0.00041326, 0.00443002]])
check("3.1 b1 zeros", b1, np.zeros((1, 4)))
check("3.1 W2", W2, [
 [0.0095155, -0.01021162, 0.00473472, -0.00267641, 0.00846771, -0.02127227, -0.00099094, -0.00602821, 0.00432263, 0.00470044],
 [-0.00707626, -0.00712174, -0.00110665, -0.00896642, 0.00841984, -0.00369207, -0.02906988, -0.00374822, -0.01038544, -0.01631276],
 [-0.01237427, 0.00109408, 0.01328641, 0.00313184, -0.00606503, 0.00455904, -0.0045909, -0.006946, -0.01154363, -0.01751829],
 [-0.00389924, 0.00158053, -0.00096624, -0.00415967, -0.00945746, 0.00608246, -0.01317132, 0.00776028, -0.01002217, -0.00752435]])
check("3.1 b2 zeros", b2, np.zeros((1, 10)))

# 3.2 relu
a = learn.relu(np.array([[-1, 0, 1], [-2, 0, 2]]))
print("relu ->", a.tolist(), "dtype:", a.dtype)
check("3.2 relu", a, [[0, 0, 1], [0, 0, 2]])

# 3.2 softmax
s = learn.softmax(np.array([[1, 2, 3], [4, 5, 6]]))
check("3.2 softmax", s, [[0.09003057, 0.24472847, 0.66524096]] * 2)
check("3.2 softmax soma por linha = 1", s.sum(axis=1), [1., 1.])

… (7 linhas omitidas)
```

<details><summary>resultado</summary>

```
shapes: (784, 4) (1, 4) (4, 10) (1, 10)
[OK] 3.1 shapes
[OK] 3.1 W1 primeiras linhas
[OK] 3.1 W1 últimas linhas
[OK] 3.1 b1 zeros
[OK] 3.1 W2
[OK] 3.1 b2 zeros
relu -> [[0, 0, 1], [0, 0, 2]] dtype: int64
[OK] 3.2 relu
[OK] 3.2 softmax
[OK] 3.2 softmax soma por linha = 1
extra softmax([[1000,1001,1002]]) -> [[nan nan nan]] | warnings: ['overflow encountered in exp', 'invalid value encountered in divide']
```

</details>

Os três testes do PDF passaram. Só que o teste extra de estabilidade da softmax falhou, e ele importa para o treino de verdade.

| Teste | Resultado |
|---|---|
| 3.1 `initialize_parameters(784, 4, 10)`: shapes, `W1` (primeiras e últimas linhas), `W2` completo, `b1` e `b2` zerados | ✅ |
| 3.2 `relu([[-1, 0, 1], [-2, 0, 2]])` → `[[0 0 1] [0 0 2]]`, ainda `int64` | ✅ |
| 3.2 `softmax([[1, 2, 3], [4, 5, 6]])` → `[0.0900 0.2447 0.6652]` nas duas linhas | ✅ |
| Extra: cada linha da softmax soma 1 (confirma que o `keepdims` está certo) | ✅ |
| Extra: `softmax([[1000, 1001, 1002]])` | ❌ `[[nan nan nan]]` |

**O problema da softmax:** ela ainda não tem o passo de estabilidade. Com logits grandes, o `np.exp` estoura, gera `inf / inf` e devolve `nan`, com os avisos `overflow encountered in exp` e `invalid value encountered in divide`. O resultado esperado é mais ou menos `[0.090, 0.245, 0.665]`, igual ao de `[1, 2, 3]`, porque só muda uma constante somada à linha. O teste do PDF não pega isso porque usa logits pequenos. No treino com taxa de aprendizado 0.1 ou 0.5, porém, o `Z2` pode crescer, e um único `nan` contamina a loss e todos os pesos. A correção é a da explicação anterior: subtrair o máximo de cada linha antes do `exp`, com `np.max(..., axis=1, keepdims=True)`.

**Detalhe de estilo:** o corpo do `relu` está indentado com 2 espaços e o resto do arquivo usa 4. Funciona, mas vale padronizar.

O script de teste está em `/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py`. Ele roda direto com `tp2-env/bin/python` e não mexe no seu `learn.py`. Quando você corrigir a softmax e implementar o `forward`, eu acrescento ao script os testes das seções 3.3 a 3.6. Os dados de entrada desses testes dependem da sequência do `np.random`, então vou encadear tudo na mesma ordem do PDF.


---

## 👤 Aluno

Teste agora.


---

## 🤖 Claude

**Ferramenta: Bash** — Show current learn.py implementation

```bash
sed -n 26,75p src/learn.py
```

<details><summary>resultado</summary>

```
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return (W1, b1, W2, b2)

# Activation functions
def relu(Z):
  return np.maximum(Z, 0)

def relu_derivative(Z):
  return Z > 0

def softmax(Z):
  normalized_z = np.max(Z, axis=1, keepdims=True)
  e_z = np.exp(normalized_z)
  return e_z / np.sum(e_z, axis=1, keepdims=True)

# Forward pass
def forward(X, parameters):
    pass

# Loss function
def cross_entropy_loss(y_true, y_pred):
    pass

# Backward pass
def backward(X, y, W2, Z1, A1, A2):
    n_samples = X.shape[0]
    
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / n_samples
    db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
    dA1 = np.dot(dZ2, W2.T)
    
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / n_samples
    db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples
    
… (7 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Rerun PDF tests and inspect softmax output

```bash
tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py 2>&1; tp2-env/bin/python -c "
import sys; sys.path.insert(0,'src'); import numpy as np, learn
Z = np.array([[1,2,3],[4,5,6]])
print('np.max(Z, axis=1, keepdims=True) =', np.max(Z, axis=1, keepdims=True).tolist())
print('softmax(Z) shape:', learn.softmax(Z).shape, '->', learn.softmax(Z).tolist())
"
```

<details><summary>resultado</summary>

```
shapes: (784, 4) (1, 4) (4, 10) (1, 10)
[OK] 3.1 shapes
[OK] 3.1 W1 primeiras linhas
[OK] 3.1 W1 últimas linhas
[OK] 3.1 b1 zeros
[OK] 3.1 W2
[OK] 3.1 b2 zeros
relu -> [[0, 0, 1], [0, 0, 2]] dtype: int64
[OK] 3.2 relu
[FALHOU] 3.2 softmax
  obtido:   [[1.]
 [1.]]
  esperado: [[0.09003057, 0.24472847, 0.66524096], [0.09003057, 0.24472847, 0.66524096]]
[OK] 3.2 softmax soma por linha = 1
extra softmax([[1000,1001,1002]]) -> [[nan]] | warnings: ['overflow encountered in exp', 'invalid value encountered in divide']
np.max(Z, axis=1, keepdims=True) = [[3], [6]]
softmax(Z) shape: (2, 1) -> [[1.0], [1.0]]
```

</details>

**Ferramenta: Bash** — Add shape and stability checks to test script

```bash
cd /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad && python3 - <<'EOF'
p = "test_pdf.py"
s = open(p).read()
s = s.replace('check("3.2 softmax soma por linha = 1", s.sum(axis=1), [1., 1.])',
 'check("3.2 softmax shape igual à entrada", s.shape == (2, 3), True)\ncheck("3.2 softmax soma por linha = 1", s.sum(axis=1), [1., 1.])')
s = s.replace('print("extra softmax([[1000,1001,1002]]) ->", big, "| warnings:", [str(x.message) for x in w])',
 'check("extra softmax estável com logits grandes", big, [[0.09003057, 0.24472847, 0.66524096]])\nprint("  warnings:", [str(x.message) for x in w])')
open(p, "w").write(s)
EOF
grep -n "shape igual\|estável" test_pdf.py
```

<details><summary>resultado</summary>

```
38:check("3.2 softmax shape igual à entrada", s.shape == (2, 3), True)
45:check("extra softmax estável com logits grandes", big, [[0.09003057, 0.24472847, 0.66524096]])
Shell cwd was reset to /home/vinicius/ufmg/tp-2-intro-ia
```

</details>

A inicialização e a ReLU continuam passando. A softmax quebrou: agora ela devolve `[[1.], [1.]]`, com shape `(2, 1)`, quando deveria devolver uma matriz `(2, 3)` de probabilidades. O `forward` ainda está com `pass`, então os testes de 3.3 em diante ainda não rodam.

| Teste | Resultado |
|---|---|
| 3.1 `initialize_parameters` | ✅ |
| 3.2 `relu` | ✅ |
| 3.2 `softmax([[1, 2, 3], [4, 5, 6]])` | ❌ sai `[[1.], [1.]]` |
| Extra: `softmax` com logits grandes | ❌ sai `[[nan]]` |

### Por que quebrou

Este é o código atual, `src/learn.py:40-43`:

```python
normalized_z = np.max(Z, axis=1, keepdims=True)
e_z = np.exp(normalized_z)
```

Seguindo com `Z = [[1, 2, 3], [4, 5, 6]]`:

1. `normalized_z` é `[[3], [6]]`. Esse é **só o máximo** de cada linha, não `Z` com o máximo subtraído.
2. `e_z` vira `[[e³], [e⁶]]`, com shape `(2, 1)`. Os valores originais de `Z` sumiram.
3. A soma de cada linha de `e_z` é o próprio elemento, então a divisão dá `1` em todas as posições.

Falta usar o máximo para **deslocar** o `Z`. O que vai para o `np.exp` precisa ter o mesmo shape de `Z`, `(2, 3)`, com o maior valor de cada linha virando 0. Com logits `[1000, 1001, 1002]`, a entrada do `exp` deve ser `[-2, -1, 0]`.

### Sobre o script de teste

Uma das minhas checagens deixou o erro passar: "soma por linha = 1" deu ✅, porque `[[1.], [1.]]` de fato soma 1 em cada linha. Acrescentei ao script uma checagem de shape e transformei o teste de estabilidade numa comparação de verdade com `[0.0900, 0.2447, 0.6652]`. Antes ele só imprimia o resultado.

Quando você corrigir, é só pedir para rodar de novo. Se o `forward` já estiver pronto nessa hora, eu incluo o teste da 3.3.


---

## 👤 Aluno

E agora? Se funcionar, me explique o que os parametros axis fazem, porque a normalizacao pelo max de Z funciona e como ela funciona. Prefira exemplos visuais.


---

## 🤖 Claude

**Ferramenta: Bash** — Show softmax and rerun PDF tests

```bash
sed -n 39,46p src/learn.py && tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py 2>&1
```

<details><summary>resultado</summary>

```
def softmax(Z):
  normalized_z = Z - np.max(Z, axis=1, keepdims=True)
  e_z = np.exp(normalized_z)
  return e_z / np.sum(e_z, axis=1, keepdims=True)

# Forward pass
def forward(X, parameters):
shapes: (784, 4) (1, 4) (4, 10) (1, 10)
[OK] 3.1 shapes
[OK] 3.1 W1 primeiras linhas
[OK] 3.1 W1 últimas linhas
[OK] 3.1 b1 zeros
[OK] 3.1 W2
[OK] 3.1 b2 zeros
relu -> [[0, 0, 1], [0, 0, 2]] dtype: int64
[OK] 3.2 relu
[OK] 3.2 softmax
[OK] 3.2 softmax shape igual à entrada
[OK] 3.2 softmax soma por linha = 1
[OK] extra softmax estável com logits grandes
  warnings: []
```

</details>

Os testes passaram, inclusive o de estabilidade: `softmax([[1000, 1001, 1002]])` agora devolve `[0.0900, 0.2447, 0.6652]` sem nenhum warning.

| Teste | Resultado |
|---|---|
| 3.1 `initialize_parameters` | ✅ |
| 3.2 `relu` | ✅ |
| 3.2 `softmax` (valores, shape e soma por linha = 1) | ✅ |
| Extra: `softmax` com logits grandes | ✅ |

---

## 1. O que o `axis` faz

Um array 2D tem dois eixos. **`axis=k` é o eixo que vai sumir** quando você agrega (com `sum`, `max`, `mean` ou `argmax`).

```
                  axis=1  →  (atravessa as colunas)
                ┌─────────────────┐
   axis=0       │  1    2    3    │   ← exemplo 0
     ↓          │  4    5    6    │   ← exemplo 1
 (desce pelas   └─────────────────┘
  linhas)          shape (2, 3)
```

**`axis=0` esmaga as linhas e deixa uma resposta por coluna:**

```
   1    2    3
   4    5    6
   ↓    ↓    ↓   np.sum(Z, axis=0)
 ───────────────
   5    7    9      shape (3,)
```

**`axis=1` esmaga as colunas e deixa uma resposta por linha:**

```
   1    2    3   →   6
   4    5    6   →  15        np.sum(Z, axis=1)
                                shape (2,)
```

**O `keepdims=True` mantém o eixo esmagado com tamanho 1 em vez de apagá-lo:**

```
 axis=1, sem keepdims:  [ 6, 15]          shape (2,)    ← virou um vetor "deitado"
 axis=1, com keepdims:  [[ 6],            shape (2, 1)  ← continua uma coluna
                         [15]]
```

Isso importa por causa do **broadcasting**. Quando você opera `(2, 3)` com `(2, 1)`, o NumPy estica a coluna para a direita até ela cobrir a matriz:

```
   Z (2,3)          max (2,1)      max esticado (2,3)       Z - max
 ┌─────────┐        ┌───┐          ┌─────────┐          ┌──────────┐
 │ 1  2  3 │   −    │ 3 │   ═══▶   │ 3  3  3 │   ═══▶   │ -2 -1  0 │
 │ 4  5  6 │        │ 6 │          │ 6  6  6 │          │ -2 -1  0 │
 └─────────┘        └───┘          └─────────┘          └──────────┘
```

Sem o `keepdims`, o NumPy tentaria alinhar `(2,)` com as **colunas** de `Z`. Isso dá erro ou, se o número de linhas for igual ao de colunas, um resultado errado sem aviso nenhum. Foi o caso da matriz 3×3 que mostrei antes.

**No seu código, cada eixo tem um papel:**

| Onde | Eixo | Pergunta que responde |
|---|---|---|
| `softmax` | `axis=1` | "Para **este exemplo**, qual é o total das 10 classes?" |
| `backward`, linha 57 (`db2`) | `axis=0` | "Para **este neurônio**, qual é a soma dos gradientes de todos os exemplos?" |

Como as linhas são exemplos e as colunas são neurônios ou classes, `axis=1` significa "dentro de cada exemplo" e `axis=0` significa "entre os exemplos". O `np.argmax` que você vai usar na acurácia segue a mesma regra.

---

## 2. Como a normalização pelo máximo funciona

Esta é a sua softmax, passo a passo, com as duas linhas do teste:

```
 Z                     [ 1     2     3   ]      [ 1000   1001   1002  ]
                              │                         │
 − max da linha        [ 1-3   2-3   3-3 ]      [ 1000-1002 ... 1002-1002 ]
                              │                         │
 normalized_z          [ -2    -1    0   ]      [ -2     -1     0     ]   ← idênticos!
                              │                         │
 exp                   [ 0.135 0.368 1.0 ]      [ 0.135  0.368  1.0   ]
                              │                         │
 ÷ soma (1.503)        [ 0.090 0.245 0.665 ]    [ 0.090  0.245  0.665 ]
```

Depois da subtração, as duas linhas viram **a mesma coisa**. A softmax só "enxerga" a **distância de cada valor até o maior**, não o valor absoluto.

### Por que o resultado não muda

Subtrair uma constante `c` no expoente equivale a dividir por `eᶜ`:

```
 e^(z − c) = eᶻ / eᶜ
```

Esse `eᶜ` aparece **em cima e embaixo** da fração, e por isso se cancela:

```
        eᶻʲ / eᶜ             eᶻʲ
 ─────────────────────  =  ────────
   Σ (eᶻᵏ / eᶜ)              Σ eᶻᵏ
```

É como calcular "% do tráfego por rota" com os valores em requisições ou em milhares de requisições. A unidade muda, mas a porcentagem fica igual.

### Por que o máximo e não outra constante

Qualquer `c` dá o mesmo resultado **na matemática**. No **computador**, o `float64` só representa até `e^709`. Subtrair o máximo garante duas coisas:

```
 depois de subtrair o max:   todos os valores ≤ 0   e   pelo menos um = 0

          ≤ 0  →  exp(≤ 0) ≤ 1          → nunca estoura para inf
          = 0  →  exp(0) = 1            → a soma é ≥ 1, nunca 0 ÷ 0
```

Isso responde a pergunta que deixei antes: por que não usar a média? Porque a média não garante que todos os valores fiquem ≤ 0.

```
 Z = [0, 2000]     média = 1000   →   Z − média = [-1000, +1000]   →   e^1000 = inf  💥
                   max   = 2000   →   Z − max   = [-2000,     0]   →   [0.0, 1.0]    ✅
```

### Armadilha

Se a diferença entre um logit e o máximo for muito grande (passar de uns 745), o `exp` desse logit vira **exatamente** `0.0` (underflow). Na softmax isso é inofensivo, porque a probabilidade era praticamente zero mesmo. Na **entropia cruzada**, que é a sua próxima função, o problema volta: `np.log(0)` dá `-inf`. Guarde isso para quando for implementá-la.

---

**Para ler depois:** a página de broadcasting do NumPy (https://numpy.org/doc/stable/user/basics.broadcasting.html). Ela tem diagramas parecidos com os de cima e as regras completas de quando os shapes são compatíveis.

Uma verificação opcional: no `forward`, `X·W1` tem shape `(m, hidden)` e `b1` tem shape `(1, hidden)`. Desenhe como o broadcasting estica o `b1` nessa soma. Ele estica na mesma direção do `max` da softmax ou na direção oposta?


---

## 👤 Aluno

Para o cross_entropy_loss, o que eu utilizaria como valor de C e m? Pra mim C é 10 pois é o numero de classes que iremos utilizar no mnist (output_size do parametro anterior). Para m, eu nao sei o que colocar.


---

## 🤖 Claude

**Ferramenta: Bash** — Find how server.py calls learn functions

```bash
grep -n "learn\.\|load_mnist\|X_train\|X_test" src/server.py | head -30
```

<details><summary>resultado</summary>

```
23:X_train, X_test = [], []
56:        parameters = learn.initialize_parameters(784, hidden_size, 10)
77:    global X_train, X_test, y_train, y_test
78:    X_train, X_test, y_train, y_test = learn.load_mnist()
81:    save_mnist_image(X_train, 0, 'mnist_digit_0.png')
159:    parameters, loss = learn.gradient_descent_step(X_train, y_train, parameters, learning_rate)
162:    y_hat = learn.forward(X_test, parameters)[-1]
163:    accuracy = learn.accuracy(y_test, y_hat)
```

</details>

Sua intuição sobre o `C` está certa para o MNIST, mas **nenhum dos dois deve ser fixado no código**. Os dois saem do shape das matrizes que a função recebe.

## De onde vêm `m` e `C`

O `y_true` e o `y_pred` têm o mesmo shape, `(m, C)`:

```
                C = colunas = classes
              ┌──────────────────────┐
   m = linhas │  uma linha por       │
   = exemplos │  exemplo avaliado    │
              └──────────────────────┘
```

- **`m` é o número de exemplos**, ou seja, o número de **linhas**: `y_true.shape[0]`.
- **`C` é o número de classes**, ou seja, o número de **colunas**: `y_true.shape[1]`.

O `m` muda conforme quem chama a função:

| Quem chama | m | C |
|---|---|---|
| Teste da seção 3.4 do PDF | 2 | **3** |
| Teste da seção 3.5 do PDF | 2 | 10 |
| Treino real (`server.py:159` passa `X_train` inteiro) | 56.000 (80% de 70.000) | 10 |

Por isso fixar `C = 10` quebraria o próprio teste do PDF, que usa 3 classes. O padrão para tirar `m` do shape já aparece no seu código, no `backward`, em `src/learn.py:53`:

```python
n_samples = X.shape[0]
```

## Você nem precisa do `C` explicitamente

A fórmula tem uma soma dupla, primeiro sobre os exemplos e depois sobre as classes. Um `np.sum` **sem `axis`** soma **todos** os elementos da matriz de uma vez, o que já cobre as duas somas. Usando os números do teste da seção 3.4:

```
 y_true             log(y_pred)                    y_true * log(y_pred)
┌─────────┐       ┌──────────────────────┐       ┌──────────────────────┐
│ 0  1  0 │   *   │ -2.303 -0.511 -1.204 │   =   │  0     -0.511   0    │
│ 1  0  0 │       │ -1.204 -0.916 -1.204 │       │ -1.204  0       0    │
└─────────┘       └──────────────────────┘       └──────────────────────┘
                                                   soma de tudo = -1.715
```

O one-hot funciona como uma **máscara**: os zeros apagam todas as classes erradas. Em cada linha sobra só o `log` da probabilidade que a rede deu para a **classe certa**. Por isso só o `m` aparece explicitamente na conta:

```
 L = −(1/m) · Σ  =  −(−1.715) / 2  =  0.8574   ✅ igual ao PDF
```

Leitura intuitiva: a perda é a **média**, entre os exemplos, de "−log da probabilidade que a rede deu à resposta certa". Se a rede dá 0.6 para a classe certa, o termo é 0.51. Se dá 0.3, é 1.20. Se desse 1.0, seria 0. Quanto mais confiante e correta a rede, mais a perda se aproxima de 0.

## Por que dividir por `m`

Sem a divisão, a perda **cresce com o tamanho do conjunto**. Com 56.000 exemplos ela seria milhares de vezes maior que com 2, mesmo com a rede igualmente boa. Dividir por `m` transforma o total numa média, e a perda fica comparável entre conjuntos de tamanhos diferentes. Esse é o mesmo motivo de o `backward` dividir os gradientes por `n_samples`.

## Armadilha

É o `log(0)` que mencionei antes. Se a softmax devolver exatamente `0.0` para a classe certa, `np.log(0)` dá `-inf` e a perda vira `inf`. Nos testes do PDF isso não acontece. No treino com taxa de aprendizado alta, pode acontecer. A solução comum é impedir que o `y_pred` chegue a zero antes do `log` (pesquise `np.clip`). Por ora, vale implementar a versão direta, passar no teste e só tratar isso se aparecer `inf` ou `nan` na curva de perda.

---

Uma verificação opcional: nesse teste, qual dos dois exemplos contribui mais para a perda, e por quê?


---

## 👤 Aluno

Implementei o cross_entropy_loss da maneira que soube fazer. Verifique sua corretude e se estiver ok, me ensine a fazer com np.sum


---

## 🤖 Claude

**Ferramenta: Bash** — Show forward and cross_entropy_loss code

```bash
sed -n 44,70p src/learn.py
```

<details><summary>resultado</summary>

```
# Forward pass
def forward(X, parameters):
    (W1, b1, W2, b2) = parameters
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return (Z1, A1, Z2, A2)

# Loss function
def cross_entropy_loss(y_true, y_pred):
    C = y_true.shape[1]
    m = y_true.shape[0]

    sum = 0
    for i in range(m):
      for c in range(C):
        sum += y_true[i][c] * np.log(y_pred[i][c])

    loss = -(1 / m) * sum
    
    return loss


# Backward pass
def backward(X, y, W2, Z1, A1, A2):
```

</details>

**Ferramenta: Bash** — Add forward/loss tests, run them, time loop vs vectorized

```bash
cd /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad && cat >> test_pdf.py <<'EOF'

# 3.3 forward (RNG continua a sequência do PDF: depois do initialize)
parameters = (W1, b1, W2, b2)
X = np.random.randn(2, 784)
Z1, A1, Z2, A2 = learn.forward(X, parameters)
check("3.3 Z1", Z1, [[0.24242511, -0.1806328, -0.32022768, 0.05051037],
                     [-0.3853982, -0.13327416, 0.02946697, 0.13882065]])
check("3.3 A1", A1, [[0.24242511, 0., 0., 0.05051037],
                     [0., 0., 0.02946697, 0.13882065]])
check("3.3 Z2", Z2, [[0.00210984, -0.00239572, 0.00109901, -0.00085894, 0.00157509, -0.00484971, -0.00090552, -0.00106941, 0.00054169, 0.00075945],
                     [-0.00090593, 0.00025165, 0.00025738, -0.00048516, -0.00149161, 0.00097871, -0.00196373, 0.00087261, -0.00173144, -0.00156075]])
check("3.3 A2", A2, [[0.10025104, 0.09980037, 0.10014976, 0.09995386, 0.10019745, 0.09955576, 0.09994921, 0.09993283, 0.10009396, 0.10011576],
                     [0.09996714, 0.10008293, 0.1000835, 0.10000921, 0.09990861, 0.10015572, 0.09986145, 0.10014509, 0.09988465, 0.0999017]])

# 3.4 cross_entropy_loss
y_true = np.array([[0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3]])
loss = learn.cross_entropy_loss(y_true, y_pred)
print("loss ->", loss)
check("3.4 cross_entropy_loss", loss, 0.8573992140459634, atol=1e-12)
EOF
cd /home/vinicius/ufmg/tp-2-intro-ia && tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py 2>&1 && tp2-env/bin/python -c "
import sys, time; sys.path.insert(0,'src'); import numpy as np, learn
m = 56000
y = np.eye(10)[np.random.randint(0,10,m)]
p = learn.softmax(np.random.randn(m,10))
t=time.perf_counter(); l1=learn.cross_entropy_loss(y,p); t1=time.perf_counter()-t
t=time.perf_counter(); l2=-np.sum(y*np.log(p))/m; t2=time.perf_counter()-t
print(f'loop: {t1:.3f}s  vetorizado: {t2*1000:.2f}ms  razão: {t1/t2:.0f}x  mesmos valores: {np.isclose(l1,l2)}')
print(f'10000 épocas só na loss com loop: {t1*10000/3600:.1f} h')
print('0 * log(0) =', 0 * np.log(np.float64(0)))
" 2>&1
```

<details><summary>resultado</summary>

```
shapes: (784, 4) (1, 4) (4, 10) (1, 10)
[OK] 3.1 shapes
[OK] 3.1 W1 primeiras linhas
[OK] 3.1 W1 últimas linhas
[OK] 3.1 b1 zeros
[OK] 3.1 W2
[OK] 3.1 b2 zeros
relu -> [[0, 0, 1], [0, 0, 2]] dtype: int64
[OK] 3.2 relu
[OK] 3.2 softmax
[OK] 3.2 softmax shape igual à entrada
[OK] 3.2 softmax soma por linha = 1
[OK] extra softmax estável com logits grandes
  warnings: []
[OK] 3.3 Z1
[OK] 3.3 A1
[OK] 3.3 Z2
[OK] 3.3 A2
loss -> 0.8573992140459634
[OK] 3.4 cross_entropy_loss
<string>:10: RuntimeWarning: divide by zero encountered in log
<string>:10: RuntimeWarning: invalid value encountered in scalar multiply
loop: 0.828s  vetorizado: 24.41ms  razão: 34x  mesmos valores: True
10000 épocas só na loss com loop: 2.3 h
0 * log(0) = nan
```

</details>

Sua versão com loop está correta: deu `0.8573992140459634`, igual ao PDF até a última casa. Aproveitei e testei também o `forward`, que você implementou nesse meio-tempo, e ele passou.

| Teste | Resultado |
|---|---|
| 3.1 `initialize_parameters` | ✅ |
| 3.2 `relu`, `softmax` e o teste extra de estabilidade | ✅ |
| 3.3 `forward` (`Z1`, `A1`, `Z2`, `A2`) | ✅ |
| 3.4 `cross_entropy_loss` | ✅ `0.8573992140459634` |

## Por que vale trocar o loop

Medi com o tamanho real do treino (56.000 × 10). O loop fica 34 vezes mais lento que a versão com `np.sum`:

```
loop:        0.828 s por chamada
vetorizado:  0.024 s por chamada        (mesmo resultado)

10.000 épocas (o padrão da interface) → 2,3 horas gastas só calculando a loss
```

## Do loop para o `np.sum`, peça por peça

O seu loop faz três coisas. Cada uma tem um equivalente vetorizado:

```
 seu loop                                         NumPy
 ───────────────────────────────────────────────  ─────────────────────────────
 for i in range(m):                    ┐
   for c in range(C):                  ├─ (3) →   np.sum(...)   sem axis
     sum += ...                        ┘
         y_true[i][c] * ...            ─── (1) →   y_true * ...   matriz inteira
         np.log(y_pred[i][c])          ─── (2) →   np.log(y_pred) matriz inteira
```

**(1) e (2): operações elemento a elemento.** É a mesma ideia da ReLU e da softmax. O `*` entre duas matrizes do mesmo shape multiplica posição por posição, e o `np.log` aplicado a uma matriz tira o log de cada célula. Os dois juntos fazem de uma vez o que o corpo do seu loop faz célula a célula:

```
 y_true            np.log(y_pred)                    produto
┌─────────┐      ┌──────────────────────┐      ┌───────────────────────┐
│ 0  1  0 │  *   │ -2.303 -0.511 -1.204 │  =   │  0     -0.511   0     │
│ 1  0  0 │      │ -1.204 -0.916 -1.204 │      │ -1.204  0       0     │
└─────────┘      └──────────────────────┘      └───────────────────────┘
                   ↑ cada célula (i, c) é o seu y_true[i][c] * np.log(y_pred[i][c])
```

Atenção: aqui o `*` é elemento a elemento, **não** é multiplicação de matrizes. Multiplicação de matrizes é o `np.dot` do `forward`. São operações diferentes.

**(3): o `np.sum` sem `axis` substitui os dois `for` e o acumulador.** Sem `axis`, ele esmaga **todos** os eixos de uma vez e devolve um único número:

```
 ┌───────────────────────┐
 │  0     -0.511   0     │
 │ -1.204  0       0     │   np.sum(M)            →  -1.715         (as duas somas da fórmula)
 └───────────────────────┘
                             np.sum(M, axis=1)    →  [-0.511, -1.204]   (só a soma interna: uma por exemplo)
```

As duas formas levam à resposta. Com `np.sum` sem `axis`, basta dividir por `m` e trocar o sinal. Com `axis=1`, você tem a perda de cada exemplo separada e pode tirar a média com `np.mean`, que já faz a divisão por `m`. A segunda forma também responde a pergunta da última vez: o segundo exemplo contribui mais (1.204 contra 0.511), porque a rede deu só 0.3 à classe certa.

Com isso, o corpo da função cai para uma ou duas linhas. Monte você, pela regra da seção 6. Depois me peça para rodar o teste de novo.

## Detalhes do código atual

- **`sum` como nome de variável** esconde a função embutida `sum()` do Python dentro da função. Aqui não quebra nada, mas é um hábito arriscado. Na versão vetorizada a variável deixa de existir.
- **Indentação:** o corpo dos `for` usa 2 espaços e o resto da função usa 4. É o mesmo ponto do `relu`.
- **Correção do que eu disse antes sobre o `log(0)`:** o problema é maior do que falei. Eu disse que só a probabilidade da classe **certa** chegar a 0 seria um problema. Na verdade, **qualquer** célula com `y_pred = 0` estraga a perda, porque `0 * log(0) = 0 * -inf = nan`, e não `0`. Rodei e confirmei: dá `nan`. Então a máscara do one-hot não protege as classes erradas. Isso vale tanto para o loop quanto para a versão com `np.sum`, e reforça o motivo para o `np.clip` se aparecer `nan` no treino.


---

## 👤 Aluno

Meu language server do python no vscode está crashando constantemente enquanto estou escrevendo código. Recebo alguns erros como 


```text

2026-11-17 19:18:42.807 [error] [
  'Starting language server, Class name = d, completed in 2043ms, has a falsy return value, , Return Value: undefined',
  [Error: Stopping the server timed out
      at /home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/vendor.bundle.js:21:9796
      at async Object.stop (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:31:11615)
      at async uj._stopClient (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43665)
      at async uj._createNewRequest (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43287)]
]
2026-11-17 19:18:42.808 [error] [
  'Restarting language server, Class name = d, completed in 2049ms, has a falsy return value, , Return Value: undefined',
  [Error: Stopping the server timed out
      at /home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/vendor.bundle.js:21:9796
      at async Object.stop (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:31:11615)
      at async uj._stopClient (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43665)
      at async uj._createNewRequest (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43287)]
]
2026-11-17 19:18:42.808 [error] [
  'Failed to restart language server, Class name = d, completed in 2049ms, has a falsy return value, , Return Value: undefined',
  [Error: Stopping the server timed out
      at /home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/vendor.bundle.js:21:9796
      at async Object.stop (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:31:11615)
      at async uj._stopClient (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43665)
      at async uj._createNewRequest (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43287)]
]

```


. O que pode seR?


---

## 🤖 Claude

**Skill: debug** — Pylance (ms-python.vscode-pylance-2026.4.1) no VS Code Remote WSL crashando/reiniciando constantemente enquanto edita src/learn.py em /home/vinicius/ufmg/tp-2-intro-ia; log: "Starting/Restarting/Failed to restart language server ... Error: Stopping the server timed out". Projeto tem venv tp2-env dentro do workspace, sem pyrightconfig/settings conhecidos.

<details><summary>resultado</summary>

```
Launching skill: debug
```

</details>

_(instruções da skill `debug` carregadas)_

**Ferramenta: Bash** — Collect memory, inotify, processes and log dirs

```bash
cd ~ && echo "== mem ==" && free -h && nproc && echo "== wslconfig ==" && (cat /mnt/c/Users/*/.wslconfig 2>/dev/null || echo "sem .wslconfig acessível") && echo "== inotify ==" && cat /proc/sys/fs/inotify/max_user_watches /proc/sys/fs/inotify/max_user_instances && echo "== processos pylance/node ==" && ps -eo pid,etime,rss,args --sort=-rss | grep -iE "pylance|server.bundle|node" | grep -v grep | head -10 | cut -c1-220 && echo "== log dirs ==" && ls -t ~/.vscode-server/data/logs/ | head -5
```

<details><summary>resultado</summary>

```
== mem ==
               total        used        free      shared  buff/cache   available
Mem:           7.7Gi       1.4Gi       5.8Gi       3.0Mi       571Mi       6.2Gi
Swap:          2.0Gi       0.0Ki       2.0Gi
12
== wslconfig ==
[wsl2]== inotify ==
524288
128
== processos pylance/node ==
  12396    01:57:46 384072 /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/node --dns-result-order=ipv4first /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/out/boots
  11990    01:57:49 136824 /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/node /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/out/server-main.js --host=127.0.0.1 --
  12040    01:57:47 74944 /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/node /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/out/bootstrap-fork --type=ptyHost --log
  10114    02:04:22 72704 /home/vinicius/.nvm/versions/node/v25.2.1/bin/node /home/vinicius/.npm/_npx/053a863a8198098a/node_modules/chrome-devtools-mcp/build/src/telemetry/watchdog/main.js --parent-pid=10103 --app-versio
  12071    01:57:47 72008 /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/node /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/out/bootstrap-fork --type=fileWatcher
  12004    01:57:48 65772 /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/node -e const net = require('net'); process.stdin.pause(); const client = net.createConnection({ host: '127.0.0.1', por
  12887    01:57:42 65248 /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/node /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/extensions/json-language-features/serve
  12020    01:57:48 62660 /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/node -e const net = require('net'); process.stdin.pause(); const client = net.createConnection({ host: '127.0.0.1', por
  12470    01:57:43 53772 /home/vinicius/.vscode-server/bin/69047191fc427f2f92828279aae1d90d8eeda7ab/node /home/vinicius/.vscode-remote-containers/dist/vscode-remote-containers-server-0.475.0.js
== log dirs ==
20261117T172230
20261117T164822
20261111T190933
20261110T104753
20261110T104630
Shell cwd was reset to /home/vinicius/ufmg/tp-2-intro-ia
```

</details>

Nenhum processo do Pylance (`server.bundle.js`) está rodando agora: o servidor está morto. Vou ver os logs da sessão atual.

**Ferramenta: Bash** — List log files in current VS Code session

```bash
cd ~/.vscode-server/data/logs/20261117T172230 && find . -type f | head -40 && echo "== arquivos com pylance/python ==" && grep -rliE "pylance|language server" . | head
```

<details><summary>resultado</summary>

```
./mcpGateway.log
./ptyhost.log
./remoteagent.log
./remoteTelemetry.log
./exthost1/remoteExtHostTelemetry.log
./exthost1/remoteexthost.log
./exthost1/vscode.github/GitHub.log
./exthost1/mechatroner.rainbow-csv/rainbow_csv_debug_channel.log
./exthost1/output_logging_20261117T172234/2-Console do Servidor Jupyter.log
./exthost1/output_logging_20261117T172234/1-Jupyter.log
./exthost1/ms-python.vscode-python-envs/Python Environments.log
./exthost1/vscode.json-language-features/Servidor de Linguagem JSON.log
./exthost1/TypeScriptTeam.jsts-chat-features/JSTS Chat Features.log
./exthost1/Anthropic.claude-code/Claude VSCode.log
./exthost1/ms-python.debugpy/Python Debugger.log
./exthost1/ms-python.python/Servidor de Linguagem Python.log
./exthost1/ms-python.python/Python.log
./exthost1/vscode.git/Git.log
== arquivos com pylance/python ==
exthost1/remoteexthost.log
exthost1/ms-python.python/Python.log
exthost1/output_logging_20261117T172234/1-Jupyter.log
exthost1/ms-python.python/Servidor de Linguagem Python.log
Shell cwd was reset to /home/vinicius/ufmg/tp-2-intro-ia
```

</details>

**Ferramenta: Bash** — Inspect Python language server log for crash causes

```bash
cd ~/.vscode-server/data/logs/20261117T172230/exthost1 && L="ms-python.python/Servidor de Linguagem Python.log"; wc -l "$L"; echo "== início =="; head -40 "$L" | cut -c1-250; echo "== contagem de restarts/crash =="; grep -ciE "restart" "$L"; grep -iE "heap|memory|out of|crash|exited|killed|SIGKILL|code [0-9]|connection.*(closed|disposed)|terminat" "$L" | sort | uniq -c | sort -rn | head -20 | cut -c1-250
```

<details><summary>resultado</summary>

```
2859 ms-python.python/Servidor de Linguagem Python.log
== início ==
2026-11-17 17:22:37.988 [info] Client: Pylance shadow experiment not started: remoteWorkspace.
2026-11-17 17:22:38.526 [info] Client: Pylance client (2026.4.1) started with python extension (2026.10.0)
2026-11-17 17:22:39.047 [info] (13222): Server root directory: file:///home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist
2026-11-17 17:22:39.305 [info] (13222): Pylance language server 2026.4.1 (pyright version 1.1.414, commit 0653de23) starting
2026-11-17 17:22:39.307 [info] (13222): Starting service instance "tp-2-intro-ia" for workspace "/home/vinicius/ufmg/tp-2-intro-ia"
2026-11-17 17:22:41.629 [info] (13222): Loading background bundle file:///home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/copilot-background.bundle.js into file:///home/vinicius/ufmg/tp-2-intro-ia...
2026-11-17 17:22:42.214 [info] (13222): PQ(tp-2-intro-ia): Priority queue background worker(2) root directory: file:///home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist
2026-11-17 17:22:42.220 [info] (13222): PQ(tp-2-intro-ia): Priority queue background worker(2) started
2026-11-17 17:22:42.291 [info] (13222): chat.disableAIFeatures is true — suppressing all AI surfaces.
2026-11-17 17:22:42.300 [info] (13222): chat.disableAIFeatures is true — suppressing all AI surfaces.
2026-11-17 17:22:42.310 [info] (13222): chat.disableAIFeatures is true — suppressing all AI surfaces.
2026-11-17 17:22:42.319 [info] (13222): chat.disableAIFeatures is true — suppressing all AI surfaces.
2026-11-17 17:22:42.346 [info] (13222): chat.disableAIFeatures is true — suppressing all AI surfaces.
2026-11-17 17:22:42.380 [info] (13222): FG(tp-2-intro-ia): Setting environmentName for service "tp-2-intro-ia": "tp2-env (3.14.10)"
2026-11-17 17:22:42.380 [info] (13222): FG(tp-2-intro-ia): Setting pythonPath for service "tp-2-intro-ia": "/home/vinicius/ufmg/tp-2-intro-ia/tp2-env/bin/python"
2026-11-17 17:22:42.381 [info] (13222): FG(tp-2-intro-ia): No include entries specified; assuming /home/vinicius/ufmg/tp-2-intro-ia
2026-11-17 17:22:42.382 [info] (13222): FG(tp-2-intro-ia): Auto-excluding **/node_modules
2026-11-17 17:22:42.382 [info] (13222): FG(tp-2-intro-ia): Auto-excluding **/__pycache__
2026-11-17 17:22:42.383 [info] (13222): FG(tp-2-intro-ia): Auto-excluding **/.*
2026-11-17 17:22:42.383 [info] (13222): FG(tp-2-intro-ia): Auto-excluding **/__editable__.*
2026-11-17 17:22:42.423 [info] (13222): FG(tp-2-intro-ia): Assuming Python version 3.14.10.final.0
2026-11-17 17:22:42.535 [info] (13222): FG(tp-2-intro-ia): Auto-excluding /home/vinicius/ufmg/tp-2-intro-ia/tp2-env
2026-11-17 17:22:42.535 [info] (13222): FG(tp-2-intro-ia): Found 3 source files
2026-11-17 17:22:42.536 [info] (13222): Settings have not changed. Skipping update.
2026-11-17 17:22:42.539 [info] (13222): Settings have not changed. Skipping update.
2026-11-17 17:22:42.544 [info] (13222): Settings have not changed. Skipping update.
2026-11-17 17:22:42.547 [info] (13222): Settings have not changed. Skipping update.
2026-11-17 17:22:43.610 [info] (13222): [Editable Installs] Starting dynamic pth file resolution for Python: /home/vinicius/ufmg/tp-2-intro-ia/tp2-env/bin/python
2026-11-17 17:22:43.614 [info] (13222): [Editable Installs] No sandbox available for Python /home/vinicius/ufmg/tp-2-intro-ia/tp2-env/bin/python. Dynamic pth file resolution requires Python 3.13 or later.
2026-11-17 17:22:43.616 [info] (13222): FG(tp-2-intro-ia): IDX(tp-2-intro-ia): Indexer background runner(3) root directory: file:///home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist (index)
2026-11-17 17:22:43.616 [info] (13222): FG(tp-2-intro-ia): IDX(tp-2-intro-ia): Indexing(3) started
2026-11-17 17:22:44.082 [info] (13222): FG(tp-2-intro-ia): IDX(tp-2-intro-ia): scanned(3) 220 files over 1 exec env
2026-11-17 17:22:44.403 [info] (13222): FG(tp-2-intro-ia): IDX(tp-2-intro-ia): indexed(3) 220 files over 1 exec env
2026-11-17 17:22:44.442 [info] (13222): FG(tp-2-intro-ia): IDX(tp-2-intro-ia): Indexing finished(3).
2026-11-17 17:37:47.232 [info] (13222): Go to Definition fell back to declaration because no source definition was found.
2026-11-17 17:37:47.444 [info] (13222): Go to Definition fell back to declaration because no source definition was found.
2026-11-17 17:39:13.786 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 1128
… (6 linhas omitidas)
```

</details>

O servidor terminou com `SIGKILL` às 19:18:45, três segundos depois das linhas que você colou. Vou ver o que veio antes.

**Ferramenta: Bash** — Show log context around the SIGKILL

```bash
cd ~/.vscode-server/data/logs/20261117T172230/exthost1 && L="ms-python.python/Servidor de Linguagem Python.log"; n=$(grep -n "SIGKILL" "$L" | cut -d: -f1); echo "linha $n"; sed -n "$((n-60)),$((n+15))p" "$L" | cut -c1-260; echo; echo "== timeline (msgs por minuto, últimos 30 min antes do kill) =="; awk '{print substr($2,1,5)}' "$L" | uniq -c | tail -25
```

<details><summary>resultado</summary>

```
linha 2859
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
2026-11-17 18:47:23.637 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
2026-11-17 18:47:23.642 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
2026-11-17 18:47:23.647 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
2026-11-17 18:47:23.655 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
2026-11-17 18:47:23.658 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
2026-11-17 18:47:23.663 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
2026-11-17 18:47:23.669 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
2026-11-17 18:47:23.673 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
2026-11-17 18:47:23.678 [info] (13222): 
*Warning*: Pylance failed to parse docstring at offset 2113

 Outputting literal text instead.
… (50 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Find what triggered the restart around 19:18

```bash
cd ~/.vscode-server/data/logs/20261117T172230/exthost1 && echo "== Python.log 19:1x ==" && grep -n "19:1[5-9]\|19:2" ms-python.python/Python.log | cut -c1-260 | head -30; echo "== Python Environments 19:1x ==" && grep -n "19:1[5-9]\|19:2" "ms-python.vscode-python-envs/Python Environments.log" | cut -c1-260 | head -30; echo "== remoteexthost 19:1x ==" && grep -n "19:1[5-9]\|19:2" remoteexthost.log | cut -c1-300 | head -30; echo "== dmesg oom ==" && (dmesg 2>&1 | grep -iE "oom|killed process" | tail -5 || true)
```

<details><summary>resultado</summary>

```
== Python.log 19:1x ==
38:2026-11-17 19:18:42.799 [error] [
46:2026-11-17 19:18:42.807 [error] [
54:2026-11-17 19:18:42.808 [error] [
62:2026-11-17 19:18:42.808 [error] [
== Python Environments 19:1x ==
100:2026-11-17 19:19:17.572 [info] Terminal activation sent: /home/vinicius/ufmg/tp-2-intro-ia/tp2-env/bin/python (succeeded)
== remoteexthost 19:1x ==
40:2026-11-17 19:18:43.809 [error] Error: Stopping the server timed out
== dmesg oom ==
Shell cwd was reset to /home/vinicius/ufmg/tp-2-intro-ia
```

</details>

**Ferramenta: Bash** — Read Python.log and exthost log around restart

```bash
cd ~/.vscode-server/data/logs/20261117T172230/exthost1 && echo "== Python.log completo (sem docstring spam) ==" && grep -v "docstring\|Outputting literal\|^$" ms-python.python/Python.log | cut -c1-260 | tail -45; echo "== remoteexthost ==" && sed -n 25,60p remoteexthost.log | cut -c1-300
```

<details><summary>resultado</summary>

```
== Python.log completo (sem docstring spam) ==
	at wrappedFn (node:internal/errors:540:14)
	at ChildProcess.exithandler (node:child_process:417:12)
	at ChildProcess.emit (node:events:526:24)
	at maybeClose (node:internal/child_process:1172:16)
	at ChildProcess._handle.onexit (node:internal/child_process:335:5)] {
  code: 127,
  killed: false,
  signal: null,
  cmd: 'C:\\Users\\vinic\\anaconda3\\python.exe -c "import sys;print(sys.executable)"'
}
2026-11-17 17:22:38.410 [info] Native locator: Refresh finished in 1696 ms
2026-11-17 17:22:39.306 [info] [test-by-project] Environment change detected; re-discovering projects for /home/vinicius/ufmg/tp-2-intro-ia
2026-11-17 17:22:39.328 [info] Iniciando servidor de idiomas Pylance.
2026-11-17 19:18:42.799 [error] [
  'Failed to start language server, Class name = p, completed in 2032ms, has a falsy return value, Arg 1: <Uri:/home/vinicius/ufmg/tp-2-intro-ia>, Arg 2: {"id":"/home/vinicius/ufmg/tp-2-intro-ia/tp2-env/bin/python","displayName":"tp2-env (3.14.10)","detailedDi
  [Error: Stopping the server timed out
  	at /home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/vendor.bundle.js:21:9796
  	at async Object.stop (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:31:11615)
  	at async uj._stopClient (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43665)
  	at async uj._createNewRequest (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43287)]
]
2026-11-17 19:18:42.807 [error] [
  'Starting language server, Class name = d, completed in 2043ms, has a falsy return value, , Return Value: undefined',
  [Error: Stopping the server timed out
  	at /home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/vendor.bundle.js:21:9796
  	at async Object.stop (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:31:11615)
  	at async uj._stopClient (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43665)
  	at async uj._createNewRequest (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43287)]
]
2026-11-17 19:18:42.808 [error] [
  'Restarting language server, Class name = d, completed in 2049ms, has a falsy return value, , Return Value: undefined',
  [Error: Stopping the server timed out
  	at /home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/vendor.bundle.js:21:9796
  	at async Object.stop (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:31:11615)
  	at async uj._stopClient (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43665)
  	at async uj._createNewRequest (/home/vinicius/.vscode-server/extensions/ms-python.vscode-pylance-2026.4.1/dist/extension.bundle.js:29:43287)]
]
2026-11-17 19:18:42.808 [error] [
  'Failed to restart language server, Class name = d, completed in 2049ms, has a falsy return value, , Return Value: undefined',
… (28 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Check Python log start and VS Code settings files

```bash
cd ~/.vscode-server/data/logs/20261117T172230/exthost1 && echo "== Python.log início ==" && grep -v "docstring\|Outputting literal\|^$" ms-python.python/Python.log | head -30 | cut -c1-260; echo "== settings ==" ; for f in ~/.vscode-server/data/Machine/settings.json /home/vinicius/ufmg/tp-2-intro-ia/.vscode/settings.json /home/vinicius/ufmg/.vscode/settings.json /mnt/c/Users/vinic/AppData/Roaming/Code/User/settings.json; do echo "--- $f"; [ -f "$f" ] && grep -nE "python|pylance|analysis|interpreter|conda" "$f" || echo "(não existe)"; done
```

**Ferramenta: Bash** — Check previous sessions for the same crash pattern

```bash
cd ~/.vscode-server/data/logs && for d in 20261117T164822 20261111T190933 20261110T104753; do f=$(find $d -name "Servidor de Linguagem Python.log" -o -name "Python Language Server.log" | head -1); echo "== $d: $f"; [ -n "$f" ] && grep -E "SIGKILL|timed out|exited|Starting service instance|heap|crash" "$f" | cut -c1-200 | head -10; done
```

<details><summary>resultado</summary>

```
== Python.log início ==
2026-11-17 17:22:36.578 [info] O experimento 'pythonSurveyNotificationcf' está ativo
2026-11-17 17:22:36.578 [info] A descoberta do ambiente é gerenciada pela extensão Ambientes do Python (ms-python.vscode-python-envs). Verifique o canal de saída "Ambientes do Python" para obter logs específicos do ambiente.
2026-11-17 17:22:36.713 [info] [test-by-project] Activating project-based testing mode
2026-11-17 17:22:36.713 [info] [test-by-project] Discovering projects for workspace: /home/vinicius/ufmg/tp-2-intro-ia
2026-11-17 17:22:36.713 [info] Native locator: Refresh started
2026-11-17 17:22:36.713 [info] [test-by-project] Found 1 total Python projects from API
2026-11-17 17:22:36.713 [info] [test-by-project] Filtered to 1 projects in workspace
2026-11-17 17:22:36.713 [info] [test-by-project] Creating adapter for: tp-2-intro-ia at file:///home/vinicius/ufmg/tp-2-intro-ia
2026-11-17 17:22:36.794 [info] > conda info --json
2026-11-17 17:22:37.559 [info] > pyenv which python
2026-11-17 17:22:37.559 [info] cwd: .
2026-11-17 17:22:37.743 [info] [test-by-project] No Python environment resolved yet for project file:///home/vinicius/ufmg/tp-2-intro-ia; deferring until one is assigned
2026-11-17 17:22:37.743 [info] [test-by-project] All adapters failed, falling back to default project
2026-11-17 17:22:37.743 [info] [test-by-project] Creating default project for: /home/vinicius/ufmg/tp-2-intro-ia
2026-11-17 17:22:37.917 [info] [test-by-project] Registered 1 project(s) for /home/vinicius/ufmg/tp-2-intro-ia
2026-11-17 17:22:37.917 [info] [test-by-project] Activated 1 project(s) for /home/vinicius/ufmg/tp-2-intro-ia
2026-11-17 17:22:37.917 [info] [test-by-project] Subscribed to Python project changes
2026-11-17 17:22:37.918 [info] [test-by-project] Subscribed to Python environment changes
2026-11-17 17:22:38.044 [info] Active interpreter [/home/vinicius/ufmg/tp-2-intro-ia]:  /home/vinicius/ufmg/tp-2-intro-ia/tp2-env/bin/python
2026-11-17 17:22:38.224 [info] > C:\Users\vinic\anaconda3\python.exe -c "import sys;print(sys.executable)"
2026-11-17 17:22:38.267 [error] [Error: Command failed: C:\Users\vinic\anaconda3\python.exe -c "import sys;print(sys.executable)"
/bin/sh: 1: C:Usersvinicanaconda3python.exe: not found
	at genericNodeError (node:internal/errors:986:15)
	at wrappedFn (node:internal/errors:540:14)
	at ChildProcess.exithandler (node:child_process:417:12)
	at ChildProcess.emit (node:events:526:24)
	at maybeClose (node:internal/child_process:1172:16)
	at ChildProcess._handle.onexit (node:internal/child_process:335:5)] {
  code: 127,
  killed: false,
== settings ==
--- /home/vinicius/.vscode-server/data/Machine/settings.json
(não existe)
--- /home/vinicius/ufmg/tp-2-intro-ia/.vscode/settings.json
(não existe)
--- /home/vinicius/ufmg/.vscode/settings.json
(não existe)
--- /mnt/c/Users/vinic/AppData/Roaming/Code/User/settings.json
3:  "python.defaultInterpreterPath": "C:\\Users\\vinic\\anaconda3\\python.exe",
… (3 linhas omitidas)
```

</details>

<details><summary>resultado</summary>

```
== 20261117T164822: 20261117T164822/exthost1/ms-python.python/Servidor de Linguagem Python.log
2026-11-17 16:48:37.657 [info] (3934): Starting service instance "tp-2-intro-ia" for workspace "/home/vinicius/ufmg/tp-2-intro-ia"
2026-11-17 17:08:35.678 [info] (3934): Starting service instance "<default>" for workspace ""
== 20261111T190933: 20261111T190933/exthost1/ms-python.python/Servidor de Linguagem Python.log
2026-11-11 19:09:49.497 [info] (6535): Starting service instance "girou" for workspace "/home/vinicius/lis/girou"
2026-11-11 19:09:51.202 [info] (6535): Starting service instance "girou" for workspace "/home/vinicius/lis/girou"
== 20261110T104753: 20261110T104753/exthost1/ms-python.python/Servidor de Linguagem Python.log
2026-11-10 10:48:04.353 [info] (58139): Starting service instance "cruzeiro-do-sul" for workspace "/home/vinicius/interviews/cruzeiro-do-sul"
2026-11-10 10:48:05.897 [info] (58139): Starting service instance "cruzeiro-do-sul" for workspace "/home/vinicius/interviews/cruzeiro-do-sul"
Shell cwd was reset to /home/vinicius/ufmg/tp-2-intro-ia
```

</details>

**Ferramenta: Bash** — Correlate file changes and env events with restart

```bash
echo "== arquivos alterados 19:15-19:20 no projeto ==" && find . -newermt "2026-11-17 19:15" ! -newermt "2026-11-17 19:20" -printf "%TT %p\n" 2>/dev/null | sort | head -20; echo "(total: $(find . -newermt '2026-11-17 19:15' ! -newermt '2026-11-17 19:20' | wc -l))"; echo "== Python Environments log (fim) ==" && tail -25 ~/.vscode-server/data/logs/20261117T172230/exthost1/ms-python.vscode-python-envs/"Python Environments.log" | cut -c1-220; echo "== linhas de log por minuto (Servidor) ==" && grep -oE "^2026-11-17 [0-9]{2}:[0-9]{2}" ~/.vscode-server/data/logs/20261117T172230/exthost1/ms-python.python/"Servidor de Linguagem Python.log" | uniq -c | sort -rn | head -8
```

<details><summary>resultado</summary>

```
== arquivos alterados 19:15-19:20 no projeto ==
19:19:04.8670376990 ./src/learn.py
(total: 1)
== Python Environments log (fim) ==
[2m   1.802517493s[0m [31mERROR[0m [2mpet::jsonrpc[0m[2m:[0m Failed to resolve env "C:\\Users\\vinic\\anaconda3\\python.exe"

2026-11-17 17:22:38.191 [warning] [priorityChain:global] defaultInterpreterPath 'C:\Users\vinic\anaconda3\python.exe' unresolvable, falling back to auto-discovery
2026-11-17 17:22:38.196 [info] Pipenv not found via settings, cache, or PATH
2026-11-17 17:22:38.196 [info] Found 0 pipenv environments
2026-11-17 17:22:38.197 [warning] No conda manager found in native finder data
2026-11-17 17:22:38.197 [warning] No conda manager found in native finder data
2026-11-17 17:22:38.203 [info] Poetry executable not found
2026-11-17 17:22:38.215 [info] Python API: Changed environment from undefined to Python 3.10.12 for: global
2026-11-17 17:22:38.215 [info] Internal: Changed environment from undefined to Python 3.10.12 for: global
2026-11-17 17:22:38.215 [info] [interpreterSelection] global: Python 3.10.12 (source: autoDiscovery)
2026-11-17 17:22:38.246 [info] Refreshing conda environments (hardRefresh=false)
2026-11-17 17:22:38.274 [info] Pipenv not found via settings, cache, or PATH
2026-11-17 17:22:38.275 [info] Refreshing pipenv environments
2026-11-17 17:22:38.277 [info] Refreshing poetry environments
2026-11-17 17:22:38.279 [info] Refreshing pyenv environments
2026-11-17 17:22:38.302 [warning] No conda manager found in native finder data
2026-11-17 17:22:38.302 [warning] No conda manager found in native finder data
2026-11-17 17:22:38.317 [info] Pipenv not found via settings, cache, or PATH
2026-11-17 17:22:38.317 [info] Found 0 pipenv environments
2026-11-17 17:22:38.318 [info] Poetry executable not found
2026-11-17 17:22:38.389 [info] Conda not found, conda features will be inactive.
2026-11-17 17:22:38.404 [info] Pipenv not found via settings, cache, or PATH
2026-11-17 17:22:38.408 [info] Environment discovery complete: 5 environments found (Global: 4, venv: 1)
2026-11-17 19:19:17.572 [info] Terminal activation sent: /home/vinicius/ufmg/tp-2-intro-ia/tp2-env/bin/python (succeeded)
== linhas de log por minuto (Servidor) ==
    514 2026-11-17 17:39
    133 2026-11-17 17:43
     57 2026-11-17 18:47
     34 2026-11-17 17:22
     13 2026-11-17 17:42
      3 2026-11-17 19:18
      2 2026-11-17 17:37
      1 2026-11-17 18:38
```

</details>

**Ferramenta: Bash** — Check kernel OOM messages and Pylance process

```bash
dmesg 2>&1 | tail -3; echo "---"; dmesg 2>/dev/null | grep -ciE "oom|killed process"; echo "--- uptime"; uptime; echo "--- pylance vivo?"; pgrep -fa "pylance.*server" | cut -c1-150 || echo "nenhum processo do Pylance"
```

<details><summary>resultado</summary>

```
[ 3960.817903] mini_init (223): drop_caches: 1
[ 6240.911255] mini_init (223): drop_caches: 1
[ 7440.890977] mini_init (223): drop_caches: 1
---
0
--- uptime
 19:21:19 up  2:35,  1 user,  load average: 0.34, 0.19, 0.11
--- pylance vivo?
38387 /bin/bash -c source /home/vinicius/.claude/shell-snapshots/snapshot-bash-1794946648397-babiqp.sh 2>/dev/null || true && shopt -u extglob 2>/dev/
```

</details>

## Diagnóstico parcial: ainda não há causa raiz confirmada

O que os logs mostram, em ordem:

1. **Isso aconteceu uma vez só.** Procurei nos logs de todas as sessões do VS Code desde 10/11. O único `SIGKILL` do Pylance é o de hoje às 19:18:45, nesta sessão, que começou às 17:22. Não há outros crashes registrados.
2. **O servidor está morto agora.** Não há processo do Pylance rodando, e por isso você não tem autocomplete nem hover desde 19:18. Ele foi morto e o restart falhou sem nova tentativa.
3. **A sequência foi esta:** às ~19:18:40 alguém pediu que o servidor (re)iniciasse para o interpretador `tp2-env` (`Failed to start language server … Arg 2: tp2-env`). Para isso o cliente precisava parar o servidor antigo. O servidor não respondeu em 2 s (`Stopping server timed out`), o cliente o matou com `SIGKILL` às 19:18:45 e desistiu. Então o `SIGKILL` foi o cliente reagindo ao timeout. Não foi o sistema matando por falta de memória: o `dmesg` não tem nenhum OOM e há 6,2 GB livres.
4. **Há um interpretador inválido configurado.** O seu `settings.json` **do Windows** (`C:\Users\vinic\AppData\Roaming\Code\User\settings.json:3`) tem `"python.defaultInterpreterPath": "C:\\Users\\vinic\\anaconda3\\python.exe"`. Essa configuração de usuário também vale dentro do WSL, onde o caminho não existe. A cada abertura aparece `/bin/sh: C:Usersvinicanaconda3python.exe: not found`, e a extensão de ambientes cai para a descoberta automática. Logo na abertura, o Pylance foi registrado 4 vezes (`DiagnosticCollection with name 'Pylance' does already exist` ×4).
5. **Nada do projeto disparou o restart.** No intervalo de 19:15 a 19:20, o único arquivo alterado no projeto foi o `src/learn.py`, às 19:19:04, depois do restart. Nada mudou na `tp2-env`. Os comandos de teste que eu rodei não aparecem como gatilho.

## Hipóteses

| # | Hipótese | Se verdadeira… | Teste | Custo |
|---|---|---|---|---|
| H1 | O servidor estava **ocupado/travado** quando o restart foi pedido. Por isso não parou em 2 s e acabou morto. | Com o log em `Trace`, aparecem análises longas (sklearn/numpy sem stubs, Python 3.14) logo antes de o editor "congelar". | Ativar o trace e reproduzir | baixo, depende de você |
| H2 | O `defaultInterpreterPath` do Windows provoca **trocas de ambiente**, e cada troca reinicia o Pylance. | Com um interpretador fixo para o WSL, os registros duplicados e os restarts somem. | Fixar o interpretador só no WSL e observar | baixo, mexe em config |
| H3 | O restart de 19:18:40 foi **manual**: você rodou "Restart Language Server" ou trocou o interpretador porque o editor tinha travado. | Nesse caso o "crash" que você vê é lentidão, e a investigação muda para H1. | Você me dizer | zero |

Não consigo seguir pelo H1 sozinho: o servidor já morreu e o log padrão não mostra o que o ocupava. Faltam duas coisas que só você tem:

1. **O que aconteceu às ~19:18:40?** Você reiniciou o servidor, trocou o interpretador ou abriu um terminal? A extensão registrou uma ativação de terminal às 19:19:17. Ou o editor já estava travado antes disso?
2. **"Crashando constantemente" na prática é o quê?** Popup de erro, autocomplete sumindo, hover que não carrega? Desde quando?

Enquanto isso, para voltar a trabalhar: `Ctrl+Shift+P` → **Developer: Reload Window**. Isso sobe o Pylance de novo. É uma mitigação, não resolve a causa.

Se quiser montar o loop de reprodução, adicione `"python.analysis.logLevel": "Trace"` e escreva código até o problema voltar. Depois me avise: eu leio o log, e o trecho logo antes do travamento deve mostrar o que estava ocupando o servidor.

Proponho já corrigir o H2, porque ele é um problema real independente do resto. Seria fixar o interpretador do projeto em `.vscode/settings.json` com `"python.defaultInterpreterPath": "${workspaceFolder}/tp2-env/bin/python"`, sem mexer no `settings.json` do Windows. Faço isso se você liberar.


---

## 👤 Aluno

Vamos resolver isso depois. Por enquanto, valide a minha cross_entropy_loss aogra.


---

## 🤖 Claude

**Ferramenta: Bash** — Show cross_entropy_loss and run PDF tests

```bash
sed -n '/def cross_entropy_loss/,/^# Backward/p' src/learn.py && tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py 2>&1 | grep -E "OK|FALHOU|loss|obtido|esperado"
```

<details><summary>resultado</summary>

```
def cross_entropy_loss(y_true, y_pred):
    C = y_true.shape[1]
    m = y_true.shape[0]

    product = y_true * np.log(y_pred)
    sum_result = np.sum(product)

    loss = -(1 / m) * sum_result

    return loss


# Backward pass
[OK] 3.1 shapes
[OK] 3.1 W1 primeiras linhas
[OK] 3.1 W1 últimas linhas
[OK] 3.1 b1 zeros
[OK] 3.1 W2
[OK] 3.1 b2 zeros
[OK] 3.2 relu
[OK] 3.2 softmax
[OK] 3.2 softmax shape igual à entrada
[OK] 3.2 softmax soma por linha = 1
[OK] extra softmax estável com logits grandes
[OK] 3.3 Z1
[OK] 3.3 A1
[OK] 3.3 Z2
[OK] 3.3 A2
loss -> 0.8573992140459634
[OK] 3.4 cross_entropy_loss
```

</details>

**Ferramenta: Bash** — Time vectorized loss at training size

```bash
tp2-env/bin/python -c "
import sys, time; sys.path.insert(0,'src'); import numpy as np, learn
m=56000; y=np.eye(10)[np.random.randint(0,10,m)]; p=learn.softmax(np.random.randn(m,10))
learn.cross_entropy_loss(y,p); t=time.perf_counter()
for _ in range(20): learn.cross_entropy_loss(y,p)
print(f'{(time.perf_counter()-t)/20*1000:.1f} ms por chamada')"
```

<details><summary>resultado</summary>

```
9.1 ms por chamada
```

</details>

A sua `cross_entropy_loss` está correta. Deu `0.8573992140459634`, igual ao PDF, e todos os testes até a 3.4 continuam passando.

| Teste | Resultado |
|---|---|
| 3.1 a 3.3 (`initialize_parameters`, `relu`, `softmax`, `forward`) | ✅ |
| 3.4 `cross_entropy_loss` | ✅ `0.8573992140459634` |

A versão com `np.sum` também ficou rápida. Com o tamanho real do treino (56.000 × 10), cada chamada leva **9 ms**, contra 828 ms da versão com loop, cerca de 90 vezes mais rápida. Em 10.000 épocas, a loss sai de 2,3 horas para uns 1,5 minuto.

Detalhes que não afetam o resultado:
- **`C` não é usado** (`C = y_true.shape[1]`). O `np.sum` sem `axis` já faz as duas somas, então a linha pode sair.
- **`log(0)`**: continua valendo o aviso da última vez. Se o `y_pred` tiver algum `0`, a loss vira `nan`. Trate isso só se aparecer na curva de perda.

Para a 3.5, falta o `gradient_descent_step`. Quando você implementar, eu acrescento ao script o teste dele com o `X` aleatório na mesma sequência do PDF, junto com o da `accuracy`.


---

## 👤 Aluno

Avalie a gradient_descent_step e se estiver ok, me ajude a entender como calculo a acurácia


---

## 🤖 Claude

**Ferramenta: Bash** — Show gradient_descent_step and accuracy code

```bash
sed -n '/def gradient_descent_step/,$p' src/learn.py | sed '/^\s*$/N;/^\s*\n\s*$/D'
```

<details><summary>resultado</summary>

```
def gradient_descent_step(X, y, parameters, learning_rate=0.01):
  (Z1, A1, Z2, A2) = forward(X, parameters)
  (W1, b1, W2, b2) = parameters

  loss = cross_entropy_loss(y, A2)
  
  (dW1, db1, dW2, db2) = backward(X, y, W2, Z1, A1, A2)

  W1 = W1 - learning_rate * dW1
  W2 = W2 - learning_rate * dW2
  b1 = b1 - learning_rate * db1
  b2 = b2 - learning_rate * db2

  return ((W1, b1, W2, b2), loss)

def accuracy(y_true, y_pred):
  pass
```

</details>

**Ferramenta: Bash** — Add 3.5/3.6 tests and run full script

```bash
cd /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad && cat >> test_pdf.py <<'EOF'

# 3.5 gradient_descent_step (novo X na sequência do RNG, parâmetros iniciais)
X = np.random.randn(2, 784)
y = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
parameters = (W1, b1, W2, b2)
(nW1, nb1, nW2, nb2), loss = learn.gradient_descent_step(X, y, parameters, learning_rate=0.01)
print("loss 3.5 ->", loss)
check("3.5 loss", loss, 2.3007908193979407, atol=1e-12)
check("3.5 W1 primeiras linhas", nW1[:3], [[0.00489277, -0.0013802, 0.00647982, 0.01522034],
                                          [-0.00232832, -0.00234474, 0.01579393, 0.00766823],
                                          [-0.00470117, 0.00545288, -0.00465534, -0.00458549]])
check("3.5 W1 últimas linhas", nW1[-3:], [[-0.00136562, 0.00739478, -0.01909403, 0.01318462],
                                         [0.00067495, -0.00411447, -0.00088931, -0.00038599],
                                         [-0.01731262, 0.01497452, 0.0003934, 0.00449739]])
check("3.5 b1", nb1, [[-4.63539605e-05, 2.45369176e-05, -1.63779507e-05, 5.55761532e-05]], atol=1e-12)
check("3.5 W2", nW2, [
 [0.00950049, -0.01007663, 0.0047197, -0.00269141, 0.00845266, -0.02128728, -0.00100588, -0.00604321, 0.00430764, 0.00468546],
 [-0.00718544, -0.0062819, -0.00121602, -0.00907558, 0.00831039, -0.00380133, -0.0291786, -0.00371489, -0.01049451, -0.01642172],
 [-0.01248018, 0.00098765, 0.01317978, 0.0030256, -0.00617091, 0.00445233, -0.00469654, -0.00599127, -0.0116493, -0.01762391],
 [-0.00409178, 0.00138703, -0.00116009, -0.00435282, -0.00964994, 0.00588846, -0.01336337, 0.00949599, -0.01021428, -0.00771637]])
check("3.5 b2", nb2, [[-0.00099922, 0.00399859, -0.0010031, -0.00100043, -0.0010005, -0.00100286, -0.00099571, 0.00399782, -0.00099769, -0.0009969]])
check("3.5 parâmetros originais intactos", np.allclose(W1[0], [0.00496714, -0.00138264, 0.00647689, 0.0152303]), True)

# 3.6 accuracy
y_true = np.array([[0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3]])
try:
    acc = learn.accuracy(y_true, y_pred)
    print("acc ->", acc)
    check("3.6 accuracy", acc if acc is not None else -1, 0.5)
except Exception as e:
    print("[FALHOU] 3.6 accuracy:", type(e).__name__, e)
EOF
cd /home/vinicius/ufmg/tp-2-intro-ia && tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py 2>&1 | grep -E "OK|FALHOU|loss|acc|obtido|esperado"
```

<details><summary>resultado</summary>

```
[OK] 3.1 shapes
[OK] 3.1 W1 primeiras linhas
[OK] 3.1 W1 últimas linhas
[OK] 3.1 b1 zeros
[OK] 3.1 W2
[OK] 3.1 b2 zeros
[OK] 3.2 relu
[OK] 3.2 softmax
[OK] 3.2 softmax shape igual à entrada
[OK] 3.2 softmax soma por linha = 1
[OK] extra softmax estável com logits grandes
[OK] 3.3 Z1
[OK] 3.3 A1
[OK] 3.3 Z2
[OK] 3.3 A2
loss -> 0.8573992140459634
[OK] 3.4 cross_entropy_loss
loss 3.5 -> 2.3007908193979407
[OK] 3.5 loss
[OK] 3.5 W1 primeiras linhas
[OK] 3.5 W1 últimas linhas
[OK] 3.5 b1
[OK] 3.5 W2
[OK] 3.5 b2
[OK] 3.5 parâmetros originais intactos
acc -> None
[FALHOU] 3.6 accuracy
  obtido:   -1
  esperado: 0.5
```

</details>

O `gradient_descent_step` está correto: todos os valores da seção 3.5 batem com o PDF.

| Teste | Resultado |
|---|---|
| 3.1 a 3.4 | ✅ |
| 3.5 `loss` = `2.3007908193979407` | ✅ |
| 3.5 `W1`, `b1`, `W2` e `b2` atualizados | ✅ |
| Extra: os parâmetros originais não foram alterados | ✅ |
| 3.6 `accuracy` | ⏳ ainda está com `pass` e devolve `None` |

Dois pontos em que você acertou sem talvez perceber:
- **A loss é calculada antes da atualização.** Ela mede os pesos que entraram na época, que é o que o PDF espera.
- **`W1 = W1 - lr * dW1` cria um array novo.** Se você tivesse escrito `W1 -= lr * dW1`, o NumPy alteraria **no lugar** o array que veio de fora, e a tupla `parameters` de quem chamou também mudaria. Aqui isso não quebraria o treino, mas é um efeito colateral que confunde testes. Foi por isso que acrescentei a checagem de "parâmetros originais intactos".

Um detalhe de estilo: a indentação está com 2 espaços de novo.

---

## Como calcular a acurácia

**O que é:** a fração dos exemplos em que a classe prevista pela rede é igual à classe verdadeira.

**Analogia:** é a "taxa de sucesso" de um painel do Grafana, `count(status == 2xx) / count(*)`. Para cada requisição você pergunta "deu certo?", e a taxa é a proporção de "sim". Aqui cada exemplo é uma requisição, e "deu certo" quer dizer "a rede chutou o dígito certo".

### O mecanismo, com o exemplo do PDF

O cálculo tem três passos.

**Passo 1: transformar cada linha num número de classe com `np.argmax(..., axis=1)`.** O `argmax` devolve o **índice** do maior valor, não o valor. Com `axis=1`, ele faz isso por linha, como na softmax:

```
 y_pred                       argmax por linha       y_true (one-hot)     argmax por linha
┌─────────────────┐                                  ┌─────────┐
│ 0.1  [0.6]  0.3 │   →   1                          │ 0 [1] 0 │   →   1
│ 0.3  [0.4]  0.3 │   →   1                          │[1] 0  0 │   →   0
└─────────────────┘                                  └─────────┘
  classe prevista: [1, 1]                              classe verdadeira: [1, 0]
```

No one-hot, o `argmax` acha a posição do `1`, que é a classe certa. É o mesmo truque nos dois lados.

**Passo 2: comparar elemento a elemento com `==`.** Entre dois arrays, o `==` devolve um array de booleanos:

```
 previsto:   [1,     1    ]
 verdadeiro: [1,     0    ]
             ──────────────
       ==    [True,  False]
```

**Passo 3: tirar a média dos booleanos com `np.mean`.** O NumPy trata `True` como 1 e `False` como 0, então a média dá exatamente a proporção de acertos:

```
 np.mean([True, False])  =  (1 + 0) / 2  =  0.5   ✅ igual ao PDF
```

Com outro exemplo, só para ver o comportamento das peças:

```python
>>> np.argmax(np.array([[5, 9, 2], [7, 1, 3]]), axis=1)
array([1, 0])
>>> np.array([2, 0, 1]) == np.array([2, 1, 1])
array([ True, False,  True])
>>> np.mean(np.array([True, False, True]))
0.6666666666666666
```

Encadeando os três passos, a função fica em uma linha, como o PDF sugere. A montagem é com você, pela seção 6.

### Armadilhas

- **`argmax` sem `axis`** achata a matriz inteira e devolve **um único índice**. Com o `y_pred` do exemplo daria `1`, a posição do 0.6 no array achatado. Aí `1 == 1` dá `True`, e a acurácia sairia 1.0 sem nenhum erro.
- **Não compare `y_pred == y_true` direto.** `0.6 == 1` é `False`, então isso compararia probabilidades com zeros e uns. O `argmax` precisa vir antes, dos dois lados.
- **Empates:** com `[0.3, 0.4, 0.3]` não há empate no máximo. Quando há, o `argmax` devolve o **primeiro** índice. No treino real, com floats, empate exato é raríssimo.

### Para o relatório: acurácia × loss

As duas medem coisas diferentes. No exemplo, a rede **acertou** o primeiro exemplo dando 0.6 à classe certa. Se tivesse dado 0.99, a acurácia seria a mesma, 0.5, e a loss cairia. A acurácia só vê se o `argmax` acertou. A loss vê **a confiança** em cada acerto ou erro. Por isso a curva de perda pode continuar caindo enquanto a de acurácia já estabilizou. A Figura 2 do PDF mostra isso a partir de mais ou menos 4.000 épocas.

**Para ler depois:** a documentação do `np.argmax` (https://numpy.org/doc/stable/reference/generated/numpy.argmax.html), que tem exemplos com `axis` e explica o comportamento em empates.

Quando implementar, me peça para rodar de novo: o teste da 3.6 já está no script. Com tudo verde, o próximo passo é subir o servidor e treinar de verdade.


---

## 👤 Aluno

Analise se a lógica está correta, se tiver altero para um one-liner


---

## 🤖 Claude

**Ferramenta: Bash** — Show accuracy code and run PDF tests

```bash
sed -n '/def accuracy/,$p' src/learn.py | sed '/^\s*$/d' && tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py 2>&1 | grep -E "FALHOU|3\.6|acc|obtido|esperado|Error"; echo "total OK: $(tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py 2>&1 | grep -c '\[OK\]')"
```

<details><summary>resultado</summary>

```
def accuracy(y_true, y_pred):
  true_classifications = np.argmax(y_true, axis=1)
  pred_classifications = np.argmax(y_pred, axis=1)
  return np.mean(true_classifications == pred_classifications)
acc -> 0.5
[OK] 3.6 accuracy
total OK: 24
```

</details>

**Ferramenta: Edit** — `/home/vinicius/ufmg/tp-2-intro-ia/src/learn.py`

Antes:

```
  true_classifications = np.argmax(y_true, axis=1)
  pred_classifications = np.argmax(y_pred, axis=1)
  return np.mean(true_classifications == pred_classifications)
```

Depois:

```
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
```

**Ferramenta: Bash** — Show new accuracy and rerun all tests

```bash
sed -n '/def accuracy/,/return/p' src/learn.py && tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py 2>&1 | grep -cE "\[OK\]" && tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/test_pdf.py 2>&1 | grep -E "FALHOU|3\.6" || true
```

<details><summary>resultado</summary>

```
The file /home/vinicius/ufmg/tp-2-intro-ia/src/learn.py has been updated successfully. (note: the file had been modified on disk since you last read it — the edit applied cleanly, but the file contains other changes not in your context. Read it before edits that depend on surrounding content.)
```

</details>

<details><summary>resultado</summary>

```
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
24
[OK] 3.6 accuracy
```

</details>

A lógica estava correta e já transformei em one-liner, em `src/learn.py`:

```python
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
```

Depois da mudança, rodei o script de novo e as 24 checagens continuam ✅, com a 3.6 dando `0.5`. Com isso, todas as funções que o PDF pede no `learn.py` passam nos "Resultados esperados" das seções 3.1 a 3.6.

O one-liner é só a sua versão de três linhas com as variáveis intermediárias embutidas. A lógica é a sua. Mesmo assim, vale citar no relatório, junto com o resto do uso de IA que a seção 6 pede para declarar. Indentei com 4 espaços, como o resto do arquivo. Nas outras funções que estão com 2 espaços (`relu`, `softmax`, `gradient_descent_step`) eu não mexi.

O próximo passo é treinar de verdade pela interface. O cálculo da loss passou de 2,3 h para cerca de 1,5 min em 10.000 épocas. O resto da época (`forward`, `backward` e as multiplicações de matrizes com 56.000 exemplos) é que vai dominar o tempo de treino. Se quiser, eu subo o servidor e acompanho as primeiras épocas para conferir se a perda cai e a acurácia sobe como na Figura 2.


---

## 👤 Aluno

Subi o servidor e botei para rodar com os parâmetros 10.000 épocas, 0.01 de taxa de aprendizado e 8 neurônios. O melhor que consegui foi Epoch: 9999, Loss: 0.3311797306880786, Accuracy: 0.9062857142857143. No entanto, eu não entendi porque ele não salvou os pesos nem os checkpoints da loss e da accuracy no meu disco do servidor. Não tem nada na pasta models e acho que deveria estar salvando. Você consegue analisar os logs do servidor para ver como esse modelo se saiu? Acho que foi satisfatório a accuracy de 90%. Mas ele ainda está errando muitas coisas no meu teste, como diferenciação entre 7 e 1, 9 ele não acertou quase nenhum, e coisas do tipo. Além de achar que demorou muito para rodar as 10.000 épocas, acho que deve ter demorado um pouco mais de 15 minutos.


---

## 🤖 Claude

**Ferramenta: Bash** — Read server.py and list model directory

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia/src && cat -n server.py && ls -la model/ && ls -la
```

<details><summary>resultado</summary>

```
1	import numpy as np
     2	import matplotlib
     3	matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
     4	import matplotlib.pyplot as plt
     5	
     6	import learn
     7	from flask import Flask, render_template, request, jsonify
     8	from flask_socketio import SocketIO, emit
     9	from flask_cors import CORS
    10	
    11	app = Flask(__name__)
    12	CORS(app)
    13	
    14	socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
    15	
    16	# RNN hiper-pamaraters
    17	epoch = 0
    18	max_epochs = 1000
    19	learning_rate = 0.01
    20	parameters = {}
    21	
    22	# Dataset variables
    23	X_train, X_test = [], []
    24	y_train, y_test = [], []
    25	
    26	# Learning curve
    27	losses = []
    28	accuracies = []
    29	
    30	@app.route('/')
    31	def index():
    32	    return render_template('index.html')
    33	
    34	@app.route('/start_learn', methods=['GET'])
    35	def start_learn():
    36	    # Get hipermarameters
    37	    hidden_size = int(request.args.get('hidden_size'))
    38	
    39	    global max_epochs
    40	    max_epochs = int(request.args.get('max_epochs'))
… (174 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Find server process cwd and stray output files

```bash
echo "== processo do servidor =="; for p in $(pgrep -f "server.py"); do echo "pid $p cwd=$(readlink /proc/$p/cwd) cmd=$(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-150)"; ls -l /proc/$p/fd 2>/dev/null | grep -E " 1 -> | 2 -> " ; done; echo "== mnist_digit_0.png / model em outros lugares =="; find /home/vinicius/ufmg -maxdepth 3 \( -name "mnist_digit_0.png" -o -name "W1.npy" -o -name "losses.csv" -o -name "model" \) -printf "%TY-%Tm-%Td %TT  %p\n" 2>/dev/null
```

<details><summary>resultado</summary>

```
== processo do servidor ==
pid 43990 cwd=/home/vinicius/ufmg/tp-2-intro-ia cmd=python3 src/server.py -port 5001 
lrwx------ 1 vinicius vinicius 64 Nov 17 19:37 1 -> /dev/pts/0
lrwx------ 1 vinicius vinicius 64 Nov 17 19:37 2 -> /dev/pts/0
pid 44392 cwd=/home/vinicius/ufmg/tp-2-intro-ia cmd=/home/vinicius/ufmg/tp-2-intro-ia/tp2-env/bin/python3 src/server.py -port 5001 
lrwx------ 1 vinicius vinicius 64 Nov 17 19:39 1 -> /dev/pts/0
lrwx------ 1 vinicius vinicius 64 Nov 17 19:39 2 -> /dev/pts/0
pid 59561 cwd=/home/vinicius/ufmg/tp-2-intro-ia/src cmd=/bin/bash -c source /home/vinicius/.claude/shell-snapshots/snapshot-bash-1794946648397-babiqp.sh 2>/dev/null || true && shopt -u extglob 2>/dev/null |
l-wx------ 1 vinicius vinicius 64 Nov 17 20:10 1 -> /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/tasks/bm98bt31j.output
l-wx------ 1 vinicius vinicius 64 Nov 17 20:10 2 -> /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/tasks/bm98bt31j.output
== mnist_digit_0.png / model em outros lugares ==
2026-11-17 19:39:19.0105418180  /home/vinicius/ufmg/tp-2-intro-ia/mnist_digit_0.png
2025-11-26 19:31:01.0000000000  /home/vinicius/ufmg/tp-2-intro-ia/src/mnist_digit_0.png
2026-10-28 17:28:02.0000000000  /home/vinicius/ufmg/tp-2-intro-ia/src/model
```

</details>

A causa dos arquivos não salvos está encontrada. Antes de responder, vou medir onde está indo o tempo de cada época e ver como a interface processa o dígito que você desenha.

**Ferramenta: Bash** — Inspect frontend for drawing preprocessing and step loop

```bash
ls static/* templates/ && grep -nE "predict|forward|28|resize|getImageData|drawImage|center|scale|socket.emit|'step'|setTimeout|setInterval" static/js/*.js 2>/dev/null | cut -c1-200 | head -50
```

**Ferramenta: Bash** — Time each part of a training epoch

```bash
ls ~/scikit_learn_data/openml 2>/dev/null | head -3; timeout 300 ../tp2-env/bin/python - <<'EOF'
import time, json, numpy as np
import learn
t=time.perf_counter(); Xtr, Xte, ytr, yte = learn.load_mnist(); print(f"load_mnist: {time.perf_counter()-t:.1f}s  X_train={Xtr.shape} dtype={Xtr.dtype}")
p = learn.initialize_parameters(784, 8, 10)
def tm(f, n=20):
    f(); t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n*1000
gd = tm(lambda: learn.gradient_descent_step(Xtr, ytr, p, 0.01))
fw = tm(lambda: learn.accuracy(yte, learn.forward(Xte, p)[-1]))
ser = tm(lambda: json.dumps([q.tolist() for q in p]))
tot = gd+fw+ser
print(f"gradient_descent_step (treino 56k): {gd:.1f} ms")
print(f"forward+accuracy (teste 14k):       {fw:.1f} ms")
print(f"serializar parâmetros p/ JSON:      {ser:.1f} ms")
print(f"soma por época: {tot:.1f} ms  -> 10.000 épocas ≈ {tot*10000/60000:.1f} min (sem contar ida e volta do socket)")
EOF
```

<details><summary>resultado</summary>

```
static/Icon
static/bootstrap.min.css
static/math.js
static/matrix.js
static/neural.js
static/notepad.js
static/output.js
static/sketch.js
static/style.css

static/images:
Icon
ufmg-logo.png

static/lib:
Icon
p5.js

templates/:
Icon
index.html
```

</details>

<details><summary>resultado</summary>

```
openml.org
<frozen site>:101: RuntimeWarning: Unexpected value in sys.prefix, expected /home/vinicius/ufmg/tp-2-intro-ia/tp2-env, got /home/vinicius/ufmg/tp-2-intro-ia/src/../tp2-env
<frozen site>:101: RuntimeWarning: Unexpected value in sys.exec_prefix, expected /home/vinicius/ufmg/tp-2-intro-ia/tp2-env, got /home/vinicius/ufmg/tp-2-intro-ia/src/../tp2-env
load_mnist: 5.2s  X_train=(56000, 784) dtype=float64
gradient_descent_step (treino 56k): 145.8 ms
forward+accuracy (teste 14k):       20.7 ms
serializar parâmetros p/ JSON:      4.5 ms
soma por época: 171.0 ms  -> 10.000 épocas ≈ 28.5 min (sem contar ida e volta do socket)
```

</details>

**Ferramenta: Bash** — Check canvas preprocessing and BLAS backend

```bash
grep -nE "28|resize|get\(|pixels|loadPixels|center|predict|feedforward|forward|step|emit" static/sketch.js static/neural.js | cut -c1-180 | head -40; ../tp2-env/bin/python -c "import numpy as np; np.show_config()" 2>/dev/null | grep -iE "name|openblas|mkl|accelerate" | head -5
```

<details><summary>resultado</summary>

```
static/sketch.js:20:    if (message['type'] == "step") {
static/sketch.js:35:        socket.emit('message_from_client',  {"type": "step"});
static/sketch.js:62:      socket.emit('message_from_client',  {"type": "step"});
static/sketch.js:85:    notepad = new Notepad(100, 130, 7, 28, 28);
      "name": "gcc",
      "name": "cython",
      "name": "gcc",
      "name": "scipy-openblas",
      "include directory": "/opt/_internal/cpython-3.14.0/lib/python3.14/site-packages/scipy_openblas64/include",
```

</details>

**Ferramenta: Bash** — Read step loop and notepad drawing logic

```bash
sed -n 15,40p static/sketch.js; grep -nE "class Notepad|getInput|toArray|pixels|blur|stroke|=\s*1\b|brush|radius|feedforward|predict" static/notepad.js static/neural.js | head -30 | cut -c1-180
```

<details><summary>resultado</summary>

```
console.log('Connected to WebSocket server');
});

// Event listener for incoming messages
socket.on('message_from_server', function (message) {    
    if (message['type'] == "step") {
        // Enable save button
        document.getElementById("saveResults").disabled = false;

        let epoch = message['epoch'];
        let loss = message['loss'];
        let accuracy = message['accuracy'];
        let parameters = message['parameters'];
        
        loadWeightsFromServer(parameters);
        
        document.getElementById("epoch").innerHTML = epoch;
        document.getElementById("loss").innerHTML = loss;
        document.getElementById("accuracy").innerHTML = accuracy;

        socket.emit('message_from_client',  {"type": "step"});
    }
    else if (message['type'] == "finish") {
        // Enable start button
        document.getElementById("startSearch").disabled = false;
    }
static/neural.js:30:      return outputs.toArray();
static/neural.js:259:        this.minLayer = 1;
static/neural.js:309:      stroke(255);
static/neural.js:310:      strokeWeight(0);
static/neural.js:343:      for (let i = 1; i < this.nLayers; i++) {
static/neural.js:382:      stroke(100);
static/neural.js:383:      strokeWeight(1);
static/neural.js:397:      stroke(clr);
static/neural.js:398:      strokeWeight(1);
static/notepad.js:1:class Notepad {
static/notepad.js:91:      for (let dX = -1; dX <= 1; dX++) {
static/notepad.js:92:        for (let dY = -1; dY <= 1; dY++) {
static/notepad.js:112:            for (let dX = -1; dX <= 1; dX++) {
static/notepad.js:113:              for (let dY = -1; dY <= 1; dY++) {
… (10 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Reproduce save failure, then rescue weights via save endpoint

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia && curl -s -m 30 http://localhost:5001/save_results; echo; mkdir model && curl -s -m 60 http://localhost:5001/save_results; echo; ls -la model/ && wc -l model/*.csv && tail -2 model/accuracies.csv model/losses.csv
```

<details><summary>resultado</summary>

```
Exit code 1
{
  "error_details": "[Errno 2] No such file or directory: 'model/W1.npy'",
  "result": "error"
}

{
  "result": "success"
}

total 80
drwxr-xr-x 2 vinicius vinicius  4096 Nov 17 20:11 .
drwxr-xr-x 6 vinicius vinicius  4096 Nov 17 20:11 ..
-rw-r--r-- 1 vinicius vinicius 50304 Nov 17 20:11 W1.npy
-rw-r--r-- 1 vinicius vinicius   768 Nov 17 20:11 W2.npy
-rw-r--r-- 1 vinicius vinicius  2500 Nov 17 20:11 accuracies.csv
-rw-r--r-- 1 vinicius vinicius   192 Nov 17 20:11 b1.npy
-rw-r--r-- 1 vinicius vinicius   208 Nov 17 20:11 b2.npy
-rw-r--r-- 1 vinicius vinicius  2500 Nov 17 20:11 losses.csv
 100 model/accuracies.csv
 100 model/losses.csv
 200 total
tail: option used in invalid context -- 2
```

</details>

Recuperei o treino: os pesos e as 100 marcações de loss e acurácia (uma a cada 100 épocas) foram salvos. Agora vou avaliar esse modelo por classe.

**Ferramenta: Bash** — Evaluate rescued model with curve and confusion matrix

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia && tp2-env/bin/python - <<'EOF' 2>&1 | grep -v RuntimeWarning
import sys; sys.path.insert(0, "src")
import numpy as np, learn
P = tuple(np.load(f"model/{n}.npy") for n in ["W1","b1","W2","b2"])
print("shapes:", [p.shape for p in P])
L = np.loadtxt("model/losses.csv"); A = np.loadtxt("model/accuracies.csv")
for i in [0,1,2,5,10,20,40,60,80,99]:
    print(f"  época {i*100:>5}: loss {L[i]:.4f}  acc {A[i]:.4f}")
Xtr, Xte, ytr, yte = learn.load_mnist()
yh = learn.forward(Xte, P)[-1]
print("acurácia teste (pesos atuais):", learn.accuracy(yte, yh))
print("loss treino (pesos atuais):", learn.cross_entropy_loss(ytr, learn.forward(Xtr, P)[-1]))
t = yte.argmax(1); p = yh.argmax(1)
cm = np.zeros((10,10), int)
np.add.at(cm, (t, p), 1)
print("\nmatriz de confusão (linha = verdadeiro, coluna = previsto)")
print("      " + " ".join(f"{j:>5}" for j in range(10)) + "   acerto")
for i in range(10):
    print(f"  {i}:  " + " ".join(f"{v:>5}" for v in cm[i]) + f"   {cm[i,i]/cm[i].sum():.1%}")
off = [(cm[i,j], i, j) for i in range(10) for j in range(10) if i!=j]
print("\n10 maiores confusões (verdadeiro -> previsto):")
for n,i,j in sorted(off, reverse=True)[:10]: print(f"  {i} -> {j}: {n} ({n/cm[i].sum():.1%} dos {i})")
print("\n7->1:", cm[7,1], " 1->7:", cm[1,7])
EOF
```

<details><summary>resultado</summary>

```
shapes: [(784, 8), (1, 8), (8, 10), (1, 10)]
  época     0: loss 2.3017  acc 0.1143
  época   100: loss 2.3003  acc 0.1143
  época   200: loss 2.2974  acc 0.1664
  época   500: loss 2.2412  acc 0.2804
  época  1000: loss 1.6705  acc 0.6425
  época  2000: loss 0.7215  acc 0.8000
  época  4000: loss 0.4354  acc 0.8809
  época  6000: loss 0.3757  acc 0.8942
  época  8000: loss 0.3491  acc 0.9027
  época  9900: loss 0.3312  acc 0.9063
acurácia teste (pesos atuais): 0.9062857142857143
loss treino (pesos atuais): 0.33117138939920593

matriz de confusão (linha = verdadeiro, coluna = previsto)
          0     1     2     3     4     5     6     7     8     9   acerto
  0:   1290     0     7     0     2    15    13     5    10     1   96.1%
  1:      0  1542     3    14     1     7     0     6    25     2   96.4%
  2:     13    25  1208    20    18     5    30    18    37     6   87.5%
  3:      3    11    31  1262     1    56    13    23    21    12   88.1%
  4:      5     4     8     5  1181     0    11     3    10    68   91.2%
  5:     15     8     4    61     7  1093    25     4    41    15   85.9%
  6:     10     8    16     1    22    23  1311     0     5     0   93.9%
  7:      6     8    24     6     8     3     0  1392     4    52   92.6%
  8:      8    27    19    42     9    48    16     7  1162    19   85.6%
  9:     10     6     7    15    53     9     0    56    17  1247   87.8%

10 maiores confusões (verdadeiro -> previsto):
  4 -> 9: 68 (5.3% dos 4)
  5 -> 3: 61 (4.8% dos 5)
  9 -> 7: 56 (3.9% dos 9)
  3 -> 5: 56 (3.9% dos 3)
  9 -> 4: 53 (3.7% dos 9)
  7 -> 9: 52 (3.5% dos 7)
  8 -> 5: 48 (3.5% dos 8)
  8 -> 3: 42 (3.1% dos 8)
  5 -> 8: 41 (3.2% dos 5)
  2 -> 8: 37 (2.7% dos 2)

7->1: 8  1->7: 6
```

</details>

**Ferramenta: Bash** — Read canvas drawing-to-input logic

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia/src/static && sed -n 80,130p notepad.js; grep -nE "notepad\.|getInputs|cells" ../static/sketch.js | head
```

<details><summary>resultado</summary>

```
this.cells[ind] += clr;
        
        overflowA = clr/2;//clr* 0.7 * 1/dist  // clr/2
        overflowB = clr/5;//clr * 0.3 * 1/dist  // clr/5
        
        if (this.overFlownCells[ind] == false) this.overflow(ind, overflowA, overflowB);
      }
    }
  
    overflow(cellIndex, flowA, flowB) {
      let indX, indY, cell;
      for (let dX = -1; dX <= 1; dX++) {
        for (let dY = -1; dY <= 1; dY++) {
          indX = this.getIndexXFromIndex(cellIndex);
          indY = this.getIndexYFromIndex(cellIndex);
  
          cell = this.getIndex(indX + dX, indY + dY);
          if (dX == 0 || dY == 0) {
            this.cells[cell] += flowA;
          } else {
            this.cells[cell] += flowB;
          }
          this.overFlownCells[cellIndex] = true;
        }
      }
    }
  
    /*
    overflow(flowA, flowB){
      let cell, indX, indY;
        for (let i = 0; i < this.cells.length; i++) {
          if (this.cells[i] >= 255 && this.overFlownCells[i] == false) {
            for (let dX = -1; dX <= 1; dX++) {
              for (let dY = -1; dY <= 1; dY++) {
                indX = this.getIndexXFromIndex(i);
                indY = this.getIndexYFromIndex(i);
                
                cell = this.getIndex(indX + dX, indY + dY);
                if (dX == 0 || dY == 0) {
                  this.cells[cell]+=flowA;
… (19 linhas omitidas)
```

</details>

## 1. Por que não salvou

**Causa raiz:** o servidor foi iniciado a partir da raiz do projeto (`python3 src/server.py`), mas o `server.py` grava com **caminho relativo** (`np.save('model/W1.npy', ...)`, em `server.py:96-103`). Esse caminho é resolvido a partir da pasta onde o processo roda. O servidor procurava `tp-2-intro-ia/model/`, que não existia. A pasta `model/` que existe fica em `src/`.

Evidências:
- O processo do servidor roda com `cwd=/home/vinicius/ufmg/tp-2-intro-ia` (lido em `/proc/<pid>/cwd`).
- Chamando o endpoint de salvar, a resposta foi `"[Errno 2] No such file or directory: 'model/W1.npy'"`.
- O `mnist_digit_0.png`, que o servidor gera ao iniciar, foi escrito às 19:39 na **raiz** do projeto, e não em `src/`.
- O erro passou calado: o salvamento automático a cada 100 épocas (`server.py:148`) cai num `try/except` que devolve um JSON de erro que ninguém lê.

**Recuperei o seu treino.** O processo continuava vivo com os pesos na memória. Criei `tp-2-intro-ia/model/` e chamei o endpoint `/save_results` do seu servidor. Saíram `W1`, `b1`, `W2` e `b2`, mais as 100 marcações de loss e acurácia. Os arquivos estão em `/home/vinicius/ufmg/tp-2-intro-ia/model/`, **na raiz**, não em `src/model/`.

**Para não acontecer de novo:** rode a partir de `src`, como os comandos do PDF assumem:
```bash
cd src && python server.py
```
O `plot.py` também lê `model/losses.csv` por caminho relativo, então rode-o de dentro de `src` também. O `-port 5001` é ignorado, porque a porta está fixa no código.

**Outra armadilha:** o servidor roda com `debug=True`, que **reinicia sozinho quando o `learn.py` muda**. O processo filho reiniciou às 19:39, no mesmo minuto em que o `learn.py` foi alterado. Se você editar o `learn.py` durante um treino, perde o estado inteiro.

## 2. Como o modelo se saiu

Não tenho acesso aos logs do servidor: eles vão só para o terminal onde você o rodou. Os pesos recuperados, porém, permitem avaliar o modelo direto. A acurácia de teste deu **90,63%**, igual ao seu log, e passa o mínimo de 90% do PDF por pouco.

**A curva de aprendizado:**
```
época     0 → loss 2.302  acc 11%    ← chute aleatório
época   500 → loss 2.241  acc 28%    ← quase parado nas primeiras ~500 épocas
época  1000 → loss 1.671  acc 64%
época  2000 → loss 0.722  acc 80%
época  4000 → loss 0.435  acc 88%
época  9900 → loss 0.331  acc 91%    ← ganhando ~0,2 ponto a cada 2.000 épocas
```

**Acerto por dígito no conjunto de teste (14.000 imagens):**
```
 0: 96.1%   1: 96.4%   2: 87.5%   3: 88.1%   4: 91.2%
 5: 85.9%   6: 93.9%   7: 92.6%   8: 85.6%   9: 87.8%
```
As maiores confusões são entre formas parecidas: 4→9 (68 casos), 5→3 (61), 9→7 (56), 3→5 (56), 9→4 (53), 7→9 (52), além de 8 com 3 e 5.

**O teste contradiz parte do que você viu na interface.** Nas imagens do MNIST, o modelo **quase não confunde 7 com 1**: foram só 8 casos de 7→1 e 6 de 1→7, em quase 3.000 imagens. E ele acerta 87,8% dos 9. Então os erros que você viu desenhando vêm principalmente de outra coisa:

- **Os seus desenhos não se parecem com o MNIST.** No MNIST, cada dígito foi redimensionado para caber numa caixa de 20×20 e centralizado pelo centro de massa. A interface manda os pixels do canvas direto para a rede (`sketch.js:121`, `normalizeArray(notepad.cells, 255)`), sem centralizar nem redimensionar. Uma rede totalmente conectada não tem invariância a translação: cada peso de `W1` está preso a um pixel fixo. Um 9 desenhado pequeno, deslocado ou com traço mais fino que o do MNIST cai em pixels que a rede quase não viu acesos durante o treino. Para comparar de forma justa, desenhe **grande e centralizado**. Isso rende uma boa discussão qualitativa para o relatório.
- **8 neurônios é pouca capacidade.** As confusões de 4↔9, 3↔5↔8 e 7↔9 são justamente entre formas que diferem em detalhes. O experimento com 16, 64 e 256 neurônios deve mostrar essas confusões caindo.

## 3. Por que demorou

Medi cada parte de uma época:

| Parte | Tempo por época | Fatia |
|---|---|---|
| `gradient_descent_step` sobre os 56.000 exemplos | 146 ms | ~85% |
| `forward` + `accuracy` sobre os 14.000 de teste | 21 ms | ~12% |
| Serializar os pesos para o navegador | 4,5 ms | ~3% |

Some a isso a ida e volta pelo socket, porque o navegador só pede a próxima época depois de receber e desenhar a anterior.

O custo vem do **gradiente descendente em lote completo**. A cada época, a matriz `X_train` inteira (56.000 × 784 em `float64`, cerca de 350 MB) passa duas vezes por multiplicações de matrizes: `X·W1` no `forward` e `Xᵀ·dZ1` no `backward`. São 10.000 épocas fazendo isso.

Isso afeta o planejamento dos experimentos:
- **O tempo cresce com o número de neurônios.** Com 256 neurônios, essas multiplicações ficam cerca de 32 vezes mais pesadas que com 8. Rodar 10.000 épocas nessa configuração levaria horas.
- **A taxa de aprendizado 0,01 é lenta.** A curva ficou quase parada nas primeiras ~500 épocas. Com 0,1 ou 0,5, a rede deve convergir em muito menos épocas. Esse é o maior ganho de tempo disponível, e o próprio experimento de taxa de aprendizado vai mostrar isso.

---

Quer que eu mova os arquivos recuperados de `model/` (na raiz) para `src/model/`? Assim eles aparecem no "Carregar Pesos" e no `plot.py` quando você rodar a partir de `src`. Também posso gerar agora os gráficos de loss e acurácia desse treino para o relatório.


---

## 👤 Aluno

mova os arquivos para src/model e gere os gráficos.


---

## 🤖 Claude

**Ferramenta: Bash** — Check destination folder and read plot.py

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia && ls -A src/model && cat -n src/plot.py
```

<details><summary>resultado</summary>

```
.DS_Store
Icon
     1	import matplotlib.pyplot as plt
     2	import numpy as np
     3	
     4	def load_data(filename):
     5	    """Load the data from the CSV file.
     6	
     7	    Args:
     8	        filename (str): Name of the CSV file to load.
     9	
    10	    Returns:
    11	        data (np.ndarray): Data loaded from the CSV file.
    12	    """
    13	    # Load the data from the CSV file
    14	    data = []
    15	    with open(filename, 'r') as file:
    16	        for line in file:
    17	            data.append([float(x) for x in line.strip().split(',')])
    18	
    19	    # Convert data to a NumPy array for easy manipulation
    20	    data_array = np.array(data)
    21	
    22	    return data_array
    23	
    24	def plot_curve(data, color, linestyle, filename, xlabel, ylabel):
    25	    """Plot the curve with the specified color and linestyle.
    26	
    27	    Args:
    28	        data (np.ndarray): Data to plot.
    29	        color (str): Color of the curve.
    30	        linestyle (str): Linestyle of the curve.
    31	        filename (str): Name of the file to save the plot.
    32	    """
    33	    # Create a new figure
    34	    plt.figure()
    35	
    36	    # Plot the data with the specified color and linestyle
    37	    plt.plot(data, color=color, linestyle=linestyle)
    38	
… (22 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Move rescued files to src/model and generate plots

```bash
mv model/W1.npy model/b1.npy model/W2.npy model/b2.npy model/losses.csv model/accuracies.csv src/model/ && rmdir model && ls -la src/model && cd src && MPLBACKEND=Agg ../tp2-env/bin/python plot.py && ls -la losses.png accuracies.png
```

<details><summary>resultado</summary>

```
total 88
drwxr-xr-x 2 vinicius vinicius  4096 Nov 17 20:15 .
drwxr-xr-x 6 vinicius vinicius  4096 Nov 17 19:37 ..
-rw-r--r-- 1 vinicius vinicius  6148 Nov 26  2025 .DS_Store
-rw------- 1 vinicius vinicius     0 Oct 28 17:28 Icon
-rw-r--r-- 1 vinicius vinicius 50304 Nov 17 20:11 W1.npy
-rw-r--r-- 1 vinicius vinicius   768 Nov 17 20:11 W2.npy
-rw-r--r-- 1 vinicius vinicius  2500 Nov 17 20:11 accuracies.csv
-rw-r--r-- 1 vinicius vinicius   192 Nov 17 20:11 b1.npy
-rw-r--r-- 1 vinicius vinicius   208 Nov 17 20:11 b2.npy
-rw-r--r-- 1 vinicius vinicius  2500 Nov 17 20:11 losses.csv
<frozen site>:101: RuntimeWarning: Unexpected value in sys.prefix, expected /home/vinicius/ufmg/tp-2-intro-ia/tp2-env, got /home/vinicius/ufmg/tp-2-intro-ia/src/../tp2-env
<frozen site>:101: RuntimeWarning: Unexpected value in sys.exec_prefix, expected /home/vinicius/ufmg/tp-2-intro-ia/tp2-env, got /home/vinicius/ufmg/tp-2-intro-ia/src/../tp2-env
-rw-r--r-- 1 vinicius vinicius 29789 Nov 17 20:15 accuracies.png
-rw-r--r-- 1 vinicius vinicius 30679 Nov 17 20:15 losses.png
```

</details>

**Ferramenta: Read** — `/home/vinicius/ufmg/tp-2-intro-ia/src/losses.png`

<details><summary>resultado</summary>

```
[imagem]
```

</details>

**Ferramenta: Read** — `/home/vinicius/ufmg/tp-2-intro-ia/src/accuracies.png`

<details><summary>resultado</summary>

```
[imagem]
```

</details>

Movi os arquivos para `src/model/` e gerei os dois gráficos com o `plot.py`, rodando de dentro de `src`:

- `/home/vinicius/ufmg/tp-2-intro-ia/src/losses.png`
- `/home/vinicius/ufmg/tp-2-intro-ia/src/accuracies.png`

Os pesos (`W1`, `b1`, `W2`, `b2`) e os CSVs agora estão em `src/model/`. A pasta `model/` que eu tinha criado na raiz ficou vazia e foi removida.

As curvas têm o mesmo formato da Figura 2 do PDF:
- **Perda:** quase parada em ~2,30 nas primeiras ~500 épocas. Cai rápido entre as épocas 500 e 2.000 e termina em 0,33, ainda descendo devagar.
- **Acurácia:** sai de 11%, passa de 64% por volta da época 1.000 e de 80% perto da época 2.000. Termina em 90,6%, com um ganho marginal nas últimas ~4.000 épocas.

Antes do próximo experimento:
- **O servidor que está rodando ainda usa a raiz do projeto como pasta de trabalho.** Pare-o e suba de novo com `cd src && python server.py`. Sem isso, o próximo treino também não salva.
- **Cada treino sobrescreve os arquivos de `src/model/` e os `.png`.** Como o relatório pede vários experimentos, copie ou renomeie os arquivos ao fim de cada um, por exemplo `losses_lr0.01_h8_e10000.csv`. Assim você não perde nenhum.


---

## 👤 Aluno

[Image #2] Coloquei um exemplo de um muito claro que eu desenhei e ele apitou como 5. Isso é uma limitação realmente do nosso modelo, do poucos neurônios que estamos utilizando, pois imaginava que com uma acurácia de 90% isso não ocorreria. Ou isso pode ser associado a alguma forma que estamos passando a imagem para o modelo interpretar? Como o arquivo server.py veio do nosso professor, eu não acho que ele estaria muito errado nessa funcionalidade que ele deu pra gente, mas queria entender sua visão.

_[imagem anexada: print da interface]_


---

## 🤖 Claude

**Ferramenta: Bash** — Read notepad indexing and weight loading code

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia/src/static && sed -n 1,80p notepad.js && sed -n 100,140p sketch.js && grep -n "function normalizeArray" -A8 *.js && grep -n "loadWeightsFromServer" -A25 sketch.js | head -40
```

<details><summary>resultado</summary>

```
class Notepad {
    constructor(x, y, cellSize, nCellsX, nCellsY) {
      this.x = x;
      this.y = y;
  
      
      this.cellSize = cellSize;
  
      this.nCellsX = nCellsX;
      this.nCellsY = nCellsY;
  
      this.w = cellSize * this.nCellsX;
      this.h = cellSize * this.nCellsY;
  
      this.img = createImage(this.nCellsX, this.nCellsY);
  
      this.clear();
    }
  
    clear() {
      this.cells = [];
      this.empty = true;
      this.overFlownCells = [];
      
      for (let i = 0; i < this.nCellsX * this.nCellsY; i++) {
        this.cells.push(0);
        this.overFlownCells.push(false);
      }
    }
  
    trackMouse(bool) {
      this.mouseTracked = bool;
      this.lastIndex = undefined;
    }
  
    update(x, y) {
      this.hover = false;
      if(this.inside(x, y)){
        let index = this.findCellIndex(x, y);
         this.hover = true;
… (130 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Read cell indexing and JS network guess function

```bash
grep -nE "getIndex\(|getIndexXFromIndex|getIndexYFromIndex|findCellIndex" -A4 notepad.js | grep -v "^--$" | head -40; echo ======; sed -n 1,80p neural.js; grep -nE "sigmoid|relu|softmax|tanh|activation" *.js | head -20
```

<details><summary>resultado</summary>

```
39:        let index = this.findCellIndex(x, y);
40-         this.hover = true;
41-        this.indexHover = index;
42-         
43-      if (this.mouseTracked) {
58:      let x0 = this.getIndexXFromIndex(currentIndex);
59:      let y0 = this.getIndexYFromIndex(currentIndex);
60:      let x1 = this.getIndexXFromIndex(this.lastIndex);
61:      let y1 = this.getIndexYFromIndex(this.lastIndex);
62-  
63-      let line = XiaolinWu.plot(x0, y0, x1, y1);
64-      let indX, indY, ind;
65-      let dy, dx, dist, clr, maxClr, overflowA, overflowB;
79:        ind = this.getIndex(indX, indY);
80-        this.cells[ind] += clr;
81-        
82-        overflowA = clr/2;//clr* 0.7 * 1/dist  // clr/2
83-        overflowB = clr/5;//clr * 0.3 * 1/dist  // clr/5
93:          indX = this.getIndexXFromIndex(cellIndex);
94:          indY = this.getIndexYFromIndex(cellIndex);
95-  
96:          cell = this.getIndex(indX + dX, indY + dY);
97-          if (dX == 0 || dY == 0) {
98-            this.cells[cell] += flowA;
99-          } else {
100-            this.cells[cell] += flowB;
114:                indX = this.getIndexXFromIndex(i);
115:                indY = this.getIndexYFromIndex(i);
116-                
117:                cell = this.getIndex(indX + dX, indY + dY);
118-                if (dX == 0 || dY == 0) {
119-                  this.cells[cell]+=flowA;
120-                }
121-                else {
132:    getIndex(indexX, indexY) {
133-      if (indexX < 0 || indexX > this.nCellsX - 1 || indexY < 0 || indexY > this.nCellsY - 1) {
134-        return undefined;
135-      } else {
136-        return indexY * this.nCellsX + indexX;
140:    getIndexXFromIndex(index) {
… (89 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Read JS guess function and activations

```bash
sed -n 85,130p neural.js; sed -n 1,12p math.js; grep -nE "softmax|outputDisplay|%" output.js | head; grep -nE "new NeuralNetwork|nn = " sketch.js
```

<details><summary>resultado</summary>

```
this.biases[i].add(gradients);
      }
    }
  
    guess(inputArray) {
      let input = inputArray;
      let output_unmapped;
      let output;
  
      // feedForward through all layers
      for (let i = 0; i < this.nLayers - 1; i++) {
        output_unmapped = this.feedForward(input, i + 1);
        output = Matrix.mapArray(output_unmapped, relu);
        input = output;
        
        if (this.visualize) { 
          if(i == this.nLayers - 2) {
            this.layerOutputs[i] = this.normalizeArray(output);
          }
          else {
            this.layerOutputs[i] = output;
          }
        }
      }
      
      return this.layerOutputs[this.layerOutputs.length-1];
    }
    
    normalizeArray(array){
       let sum = 0;
      for(let i = 0; i < array.length; i++){
        sum+= array[i];
      }
    let result = [];
      for(let i = 0; i < array.length; i++){
       result.push(array[i]/sum) 
      }
      return result;
    }
  
… (22 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Locate the drawing grid border in the screenshot

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia && tp2-env/bin/python - <<'EOF' 2>&1 | grep -v RuntimeWarning
from PIL import Image
import numpy as np
im = np.asarray(Image.open("/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/images/2.png").convert("L")).astype(int)
print(im.shape)
# procura a borda cinza do quadro do notepad (região x 100-340, y 350-590)
sub = im[350:590, 100:340]
rows = [y+350 for y in range(sub.shape[0]) if ((sub[y] > 60) & (sub[y] < 140)).sum() > 150]
cols = [x+100 for x in range(sub.shape[1]) if ((sub[:,x] > 60) & (sub[:,x] < 140)).sum() > 150]
print("linhas da borda:", rows, "colunas da borda:", cols)
EOF
```

<details><summary>resultado</summary>

```
(785, 887)
linhas da borda: [] colunas da borda: []
```

</details>

**Ferramenta: Bash** — Inspect pixel profiles around grid border

```bash
tp2-env/bin/python - <<'EOF' 2>&1 | grep -v RuntimeWarning
from PIL import Image
import numpy as np
im = np.asarray(Image.open("/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/images/2.png").convert("L")).astype(int)
y = 470
print("linha y=470, x 110-140:", im[y,110:140].tolist())
print("linha y=470, x 305-335:", im[y,305:335].tolist())
x = 220
print("coluna x=220, y 360-390:", im[360:390,x].tolist())
print("coluna x=220, y 555-585:", im[555:585,x].tolist())
EOF
```

<details><summary>resultado</summary>

```
linha y=470, x 110-140: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
linha y=470, x 305-335: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
coluna x=220, y 360-390: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 70, 70]
coluna x=220, y 555-585: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

</details>

**Ferramenta: Bash** — Reconstruct drawn digit from screenshot and predict

```bash
mkdir -p /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad && cat > /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/drawing.py <<'EOF'
import sys; sys.path.insert(0, "/home/vinicius/ufmg/tp-2-intro-ia/src")
import numpy as np
from PIL import Image
from scipy import ndimage
import learn

SHOT = "/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/images/2.png"
im = np.asarray(Image.open(SHOT).convert("L")).astype(float)
x0, x1, y0, y1 = 124.5, 321.0, 372.5, 569.0
cw, ch = (x1 - x0) / 28, (y1 - y0) / 28
grid = np.zeros((28, 28))
for r in range(28):
    for c in range(28):
        cy, cx = int(y0 + (r + .5) * ch), int(x0 + (c + .5) * cw)
        grid[r, c] = np.median(im[cy-1:cy+2, cx-1:cx+2])
x_draw = grid / 255.0

P = tuple(np.load(f"/home/vinicius/ufmg/tp-2-intro-ia/src/model/{n}.npy") for n in ["W1", "b1", "W2", "b2"])

def predict(img):
    Z1, A1, Z2, A2 = learn.forward(img.reshape(1, 784), P)
    js = np.maximum(Z2[0], 0); js = js / js.sum() if js.sum() > 0 else js
    return A2[0], js

def show(name, img):
    sm, js = predict(img)
    top = np.argsort(sm)[::-1][:3]
    print(f"{name:<38} previsão={sm.argmax()}  softmax top3: " + ", ".join(f"{k}:{sm[k]:.1%}" for k in top))
    return sm, js

def ascii(img):
    for row in img:
        print("   " + "".join(" .:-=+*#%@"[min(9, int(v * 9.99))] for v in row))

def mnist_style(img):
    ys, xs = np.nonzero(img > 0.05)
    crop = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape; s = 20 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
… (14 linhas omitidas)
```

<details><summary>resultado</summary>

```
seu desenho reconstruído do print (28x28):
                               
                               
               .:.             
               -@-             
               +@+             
               #@#             
              .#@=             
              +@@:             
             +@@@=             
           .*@@@@#             
          .+@@#%@+             
         .+@*-=@@-             
         :@+. +@#              
         .:.  *@*              
              *@*              
              #@#              
              +@+              
              =@=              
              +@+              
              #@#              
              +@+              
              =@=              
              -@@              
              .:.              
                               
                               
                               
                               
desenho como veio do canvas            previsão=5  softmax top3: 5:67.7%, 8:30.7%, 0:0.8%
  % que a interface mostraria (relu+normaliza): {0: '11.1%', 5: '42.6%', 6: '9.3%', 8: '37.0%'}
  interface no print:                           0:12.4%, 2:1.1%, 5:39.2%, 6:8.1%, 8:39.1%
```

</details>

**Ferramenta: Bash** — Train a 64-neuron model in background for comparison

```bash
cat > /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/train64.py <<'EOF'
import sys, time; sys.path.insert(0, "/home/vinicius/ufmg/tp-2-intro-ia/src")
import numpy as np, learn
out = "/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/m64"
Xtr, Xte, ytr, yte = learn.load_mnist()
p = learn.initialize_parameters(784, 64, 10)
t = time.perf_counter()
for e in range(1000):
    p, loss = learn.gradient_descent_step(Xtr, ytr, p, 0.5)
    if e % 100 == 0 or e == 999:
        print(f"época {e}: loss {loss:.4f} acc {learn.accuracy(yte, learn.forward(Xte, p)[-1]):.4f} ({time.perf_counter()-t:.0f}s)", flush=True)
np.savez(out, *p)
EOF
nohup tp2-env/bin/python /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/train64.py > /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/train64.log 2>&1 &
echo started
```

**Ferramenta: Bash** — Test drawing variants: no flag, MNIST-style preprocessing

```bash
tp2-env/bin/python -c "
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad')
from drawing import *
Xtr, Xte, ytr, yte = learn.load_mnist()
# H: estilo do 1 — o seu tem 'bandeirinha'; como são os 1 do MNIST?
ones = Xte[yte.argmax(1) == 1]
print('1s do MNIST de teste:', len(ones))
mean1 = ones.mean(0).reshape(28, 28)
print('média dos 1 do MNIST:'); ascii(mean1 / mean1.max())
show('média dos 1 do MNIST', mean1 / mean1.max())
# sem a bandeirinha: apaga a parte esquerda (colunas < 13) do seu desenho
noflag = x_draw.copy(); noflag[:, :13] = 0
show('seu desenho SEM a bandeirinha', noflag)
# estilo MNIST (20x20 + centro de massa)
ms = mnist_style(x_draw)
print('seu desenho no padrão MNIST (20x20 + centro de massa):'); ascii(ms)
show('seu desenho no padrão MNIST', ms)
show('sem bandeirinha + padrão MNIST', mnist_style(noflag))
# quantos 1 do MNIST têm traço à esquerda do topo (bandeirinha)? proxy: massa fora da coluna principal
" 2>&1 | grep -v "RuntimeWarning\|frozen"
```

<details><summary>resultado</summary>

```
started
```

</details>

<details><summary>resultado</summary>

```
1s do MNIST de teste: 1600
média dos 1 do MNIST:
                               
                               
                               
                               
                ......         
               .:----:..       
               .:===--:.       
               .:=++=-..       
               .:+*+=:.        
               .-+#*=:.        
               .-*%*-.         
               .=#%+:          
               .+@%=.          
               .*@%-           
               :#@#:           
               -%@*.           
              .=%@+.           
              :+%#=.           
             .-+#*-.           
            .:=+*+-.           
            .:=+*+-.           
           .:-=++=-.           
           .:-====-.           
           ..:----:.           
                               
                               
                               
                               
média dos 1 do MNIST                   previsão=1  softmax top3: 1:98.4%, 8:0.8%, 3:0.4%
seu desenho SEM a bandeirinha          previsão=1  softmax top3: 1:42.8%, 3:21.4%, 5:14.1%
seu desenho no padrão MNIST (20x20 + centro de massa):
                               
                               
                               
                               
                               
                  :.           
                 :@+           
… (23 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Test translation sensitivity on drawing and MNIST test set

```bash
tp2-env/bin/python -c "
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad')
from drawing import *
cy, cx = ndimage.center_of_mass(x_draw); print(f'centro de massa do seu desenho: linha {cy:.1f}, coluna {cx:.1f}  (MNIST: 14, 14)')
for dx in [0, 1, 2, 3, 4]:
    show(f'seu desenho deslocado {dx}px p/ direita', np.roll(x_draw, dx, axis=1))
sh = ndimage.shift(x_draw, (14-cy, 14-cx), order=1)
show('só centralizado (centro de massa)', sh)
Xtr, Xte, ytr, yte = learn.load_mnist()
print()
print('acurácia no MNIST de teste com as imagens deslocadas (modelo de 8 neurônios):')
imgs = Xte.reshape(-1, 28, 28)
for d in [0, 1, 2, 3, 4]:
    Xs = np.roll(imgs, d, axis=2).reshape(-1, 784)
    print(f'  {d}px p/ direita: {learn.accuracy(yte, learn.forward(Xs, P)[-1]):.1%}')
" 2>&1 | grep -v "RuntimeWarning\|frozen"; echo; cat /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/train64.log | grep -v "Runtime\|frozen"
```

<details><summary>resultado</summary>

```
centro de massa do seu desenho: linha 11.9, coluna 11.7  (MNIST: 14, 14)
seu desenho deslocado 0px p/ direita   previsão=5  softmax top3: 5:67.7%, 8:30.7%, 0:0.8%
seu desenho deslocado 1px p/ direita   previsão=8  softmax top3: 8:51.1%, 1:27.2%, 5:20.6%
seu desenho deslocado 2px p/ direita   previsão=1  softmax top3: 1:94.9%, 7:1.9%, 8:1.9%
seu desenho deslocado 3px p/ direita   previsão=7  softmax top3: 7:72.9%, 1:11.4%, 3:8.3%
seu desenho deslocado 4px p/ direita   previsão=7  softmax top3: 7:69.4%, 9:16.2%, 3:13.5%
só centralizado (centro de massa)      previsão=1  softmax top3: 1:78.6%, 3:7.4%, 8:4.4%

acurácia no MNIST de teste com as imagens deslocadas (modelo de 8 neurônios):
  0px p/ direita: 90.6%
  1px p/ direita: 84.9%
  2px p/ direita: 64.0%
  3px p/ direita: 41.8%
  4px p/ direita: 21.7%

época 0: loss 2.3027 acc 0.1452 (0s)
época 100: loss 0.3364 acc 0.9044 (23s)
```

</details>

O deslocamento já explica o 5. Falta ver o que mais neurônios mudam, para separar as duas hipóteses. Vou esperar o modelo de 64 neurônios terminar.

**Ferramenta: Bash** — Wait for 64-neuron training to finish

```bash
F=/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/m64.npz; timeout 400 bash -c "until [ -f $F ]; do sleep 5; done"; grep -v "Runtime\|frozen" /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/train64.log
```

<details><summary>resultado</summary>

```
época 0: loss 2.3027 acc 0.1452 (0s)
época 100: loss 0.3364 acc 0.9044 (23s)
época 200: loss 0.2580 acc 0.9246 (46s)
época 300: loss 0.2135 acc 0.9361 (66s)
época 400: loss 0.1826 acc 0.9434 (86s)
época 500: loss 0.1599 acc 0.9498 (106s)
época 600: loss 0.1423 acc 0.9539 (126s)
época 700: loss 0.1281 acc 0.9569 (148s)
época 800: loss 0.1165 acc 0.9591 (168s)
época 900: loss 0.1069 acc 0.9611 (188s)
época 999: loss 0.0987 acc 0.9630 (207s)
```

</details>

**Ferramenta: Bash** — Evaluate drawing and shifts on the 64-neuron model

```bash
tp2-env/bin/python -c "
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad')
import drawing
from drawing import *
z = np.load('/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/m64.npz')
drawing.P = tuple(z[f'arr_{i}'] for i in range(4))
print('== modelo 64 neurônios, lr 0.5, 1000 épocas (acc teste 96.3%) ==')
for dx in [0, 1, 2, 3, 4]:
    show(f'seu desenho deslocado {dx}px p/ direita', np.roll(x_draw, dx, axis=1))
show('seu desenho no padrão MNIST', mnist_style(x_draw))
Xtr, Xte, ytr, yte = learn.load_mnist()
imgs = Xte.reshape(-1, 28, 28)
print('acurácia MNIST teste deslocado:', '  '.join(f'{d}px: {learn.accuracy(yte, learn.forward(np.roll(imgs, d, axis=2).reshape(-1,784), drawing.P)[-1]):.1%}' for d in range(5)))
" 2>&1 | grep -v "RuntimeWarning\|frozen"
```

<details><summary>resultado</summary>

```
== modelo 64 neurônios, lr 0.5, 1000 épocas (acc teste 96.3%) ==
seu desenho deslocado 0px p/ direita   previsão=8  softmax top3: 8:84.8%, 5:8.4%, 2:3.2%
seu desenho deslocado 1px p/ direita   previsão=1  softmax top3: 1:75.3%, 8:20.3%, 5:1.7%
seu desenho deslocado 2px p/ direita   previsão=1  softmax top3: 1:99.3%, 9:0.4%, 8:0.2%
seu desenho deslocado 3px p/ direita   previsão=1  softmax top3: 1:81.3%, 9:12.2%, 7:4.2%
seu desenho deslocado 4px p/ direita   previsão=9  softmax top3: 9:34.1%, 3:30.2%, 7:25.6%
seu desenho no padrão MNIST            previsão=1  softmax top3: 1:95.3%, 4:3.3%, 9:0.8%
acurácia MNIST teste deslocado: 0px: 96.3%  1px: 92.6%  2px: 74.8%  3px: 53.7%  4px: 33.4%
```

</details>

## Diagnóstico

**Causa raiz:** o seu desenho chega ao modelo **cerca de 2 pixels deslocado** em relação ao padrão em que os dígitos do MNIST foram treinados. Uma rede totalmente conectada é muito sensível a isso. A culpa principal não é do número de neurônios, e o código do professor também não está errado.

### O que testei

Reconstruí do seu print a grade de 28×28 que foi para a rede. A reconstrução é fiel: em Python, o modelo também responde **5**, com porcentagens a até 4 pontos das que aparecem na tela. Depois mudei uma variável por vez:

| Entrada | 8 neurônios (90,6%) | 64 neurônios (96,3%)* |
|---|---|---|
| Seu desenho como veio do canvas | **5** (68%) | **8** (85%) |
| Deslocado 1 px para a direita | 8 | 1 (75%) |
| Deslocado 2 px para a direita | **1 (95%)** | **1 (99%)** |
| Deslocado 3 px para a direita | 7 | 1 (81%) |
| Redimensionado e centralizado como no MNIST | **1 (89%)** | **1 (95%)** |
| Sem a "bandeirinha" do topo, sem mover | 1 (43%, chute fraco) | — |

\*Treinei esse modelo de 64 neurônios à parte para comparar (lr 0,5, 1.000 épocas, ~3,5 min). Ele **não** sobrescreveu nada em `src/model/`.

Repeti o mesmo deslocamento em **todo o conjunto de teste do MNIST**:

| Deslocamento | 0 px | 1 px | 2 px | 3 px | 4 px |
|---|---|---|---|---|---|
| 8 neurônios | 90,6% | 84,9% | **64,0%** | 41,8% | 21,7% |
| 64 neurônios | 96,3% | 92,6% | **74,8%** | 53,7% | 33,4% |

### Por que isso acontece

- **O MNIST é padronizado.** Cada dígito foi redimensionado para caber numa caixa de 20×20 e centralizado pelo **centro de massa** na posição (14, 14). O centro de massa do seu desenho está em **(11,9; 11,7)**: uns 2 pixels acima e à esquerda. A "bandeirinha" ajuda a puxar o centro para a esquerda, e o traço ocupa 22 linhas em vez de 20.
- **Cada peso de `W1` está preso a um pixel fixo.** A rede aprendeu "o traço do 1 passa *por estas* colunas". Ela não tem noção de "um traço vertical em qualquer lugar". Se o traço anda 2 colunas, ele acende pixels que, no treino, eram típicos de 5 e 8. É por isso que um deslocamento invisível ao olho troca a resposta.
- **Os 90% valem para dados com a mesma distribuição do treino.** A acurácia de teste mede o desempenho em imagens padronizadas do mesmo jeito. O seu desenho vem de outra distribuição (*distribution shift*).

### E os neurônios?

Eles ajudam, mas não resolvem. Com 64 neurônios, a faixa de deslocamento tolerada aumenta: o 1 é reconhecido de 1 a 3 px, contra só 2 px com 8 neurônios. Mesmo assim, **o modelo de 96% também erra o seu desenho original**, dizendo 8. As confusões entre formas parecidas (4↔9, 3↔5↔8) é que caem com mais capacidade.

### Sobre o código do professor

O `server.py` não mexe no desenho. A previsão é feita no navegador (`static/sketch.js:121`), que manda os pixels do canvas direto para a rede, sem centralizar nem redimensionar. É uma simplificação razoável para uma ferramenta didática, não um bug. Um detalhe à parte: as porcentagens da tela **não são a softmax**. O JavaScript aplica ReLU na saída e divide pela soma (`neural.js:97`). A classe escolhida é a mesma, mas os números não são probabilidades. Por isso aparecem vários "0%".

## Caminhos

1. **Para usar a interface agora (recomendado):** desenhe pensando no **centro de massa**, não no centro visual. Deixe uns 4 quadradinhos de margem em cima e embaixo e compense elementos como a bandeirinha. Isso deve resolver boa parte dos casos de 7×1 e dos 9.
2. **Para o relatório:** isso dá uma ótima discussão qualitativa. A tabela de deslocamento do MNIST quantifica a limitação. Se usar esses dados, declare-os como gerados com IA, pela seção 6.
3. **Fora do escopo do TP:** centralizar o desenho no front-end ou treinar com imagens deslocadas (*data augmentation*). As duas opções mexem em coisas que o PDF não pede, e a segunda altera o `load_mnist`. Eu só faria com o aval do professor.


---

## 👤 Aluno

Ok, o seu treinamento do modelo com 64 neurônios rodou bem mais rápido que o meu por algum motivo. Assim, agora quero que você realize os treinamentos para preencher o que é necessário para o relatório. Analise o que precisa preparar para isso, me explique porque o seu rodou bem mais rápido. E vamos rodar. As alterações de parâmetros que o relatório pede, salvando o que pudermos, como se estivéssemos rodando pela interface.


---

## 🤖 Claude

Antes de rodar, uma correção: o meu treino **não foi mais rápido por época**. Ele foi mais lento. Comparando:

| | Seu treino (interface) | Meu treino (script) |
|---|---|---|
| Neurônios | 8 | 64 |
| Taxa de aprendizado | 0,01 | **0,5** |
| Épocas | 10.000 | **1.000** |
| Tempo por época | ~90 ms | ~207 ms |
| Tempo total | ~15 min | ~3,5 min |
| Acurácia | 90,6% | 96,3% |

Terminou antes porque precisou de **10 vezes menos épocas**. Com taxa de aprendizado 0,5, cada passo do gradiente anda 50 vezes mais que com 0,01. A rede já passa de 90% por volta da época 100, quando a sua ainda estava travada em 2,30 de perda. Uma época com 64 neurônios custa mais que com 8, e a interface ainda soma o envio dos pesos pelo socket e o redesenho a cada época. O que fez diferença foi a taxa de aprendizado, que é justamente o primeiro experimento do relatório.

## O plano

A seção 4 do PDF pede três experimentos, cada um variando um hiperparâmetro com os outros fixos, e depois a melhor configuração (≥ 90%) com as suas curvas. Vou fazer assim:

1. **Taxa de aprendizado** em 0,001, 0,01, 0,1 e 0,5, com 64 neurônios e 500 épocas fixos.
2. **Neurônios** em 4, 16, 64 e 256, com a melhor taxa do passo 1 e 500 épocas.
3. **Épocas** em 50, 100, 200 e 500, com a melhor taxa e 64 neurônios.
4. **Melhor configuração:** melhor taxa, melhor número de neurônios e 500 épocas, com as curvas.

Para ficar como a interface:
- Cada treino usa as funções do seu `learn.py` na mesma ordem do `server.py`: `initialize_parameters`, depois `gradient_descent_step` sobre o treino e `accuracy` sobre o teste a cada época.
- Os arquivos saem no mesmo formato: `W1`/`b1`/`W2`/`b2.npy`, mais `losses.csv` e `accuracies.csv` com um ponto a cada 100 épocas.
- **Gravo também as curvas época a época.** No formato da interface, um treino de 50 épocas não gera **nenhum** ponto, porque o servidor só registra quando a época chega a um múltiplo de 100.
- Fixo a seed (42) antes de cada inicialização, para os treinos serem reproduzíveis e comparáveis. A interface não faz isso.

A seção 6 do PDF permite usar IA para "scripts para executar os experimentos e coletar métricas", desde que isso seja declarado. O script fica em `experiments/` no projeto, para você citar.

**Ferramenta: Write** — `/home/vinicius/ufmg/tp-2-intro-ia/experiments/run_experiments.py`

<details><summary>conteúdo</summary>

```
"""Roda os experimentos da seção 4 do TP2 usando as funções de src/learn.py.

Reproduz o laço do server.py: a cada época, um gradient_descent_step no conjunto
de treino e a acurácia no conjunto de teste. Para cada treino salva, em
experiments/results/<nome>/:
  - W1.npy, b1.npy, W2.npy, b2.npy  (pesos finais, como o "Salvar Pesos")
  - losses.csv, accuracies.csv      (formato da interface: 1 ponto a cada 100 épocas)
  - losses_full.csv, accuracies_full.csv (1 ponto por época)
  - losses.png, accuracies.png      (mesmo estilo do plot.py, por época)
e acumula uma linha em experiments/results/summary.csv.

Uso: python experiments/run_experiments.py   (a partir da raiz do projeto)
"""
import csv
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import learn  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results")
SEED = 42
BASE_HIDDEN = 64
BASE_EPOCHS = 500

X_train, X_test, y_train, y_test = learn.load_mnist()
cache = {}


def plot(values, color, path, xlabel, ylabel):
    plt.figure()
    plt.plot(values, color=color, linestyle="solid")
    plt.grid(True, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()


def train(lr, hidden, epochs):
    key = (lr, hidden, epochs)
    if key in cache:
        print(f"   (reaproveitando lr={lr} h={hidden} e={epochs})", flush=True)
        return cache[key]

    name = f"lr{lr}_h{hidden}_e{epochs}"
    out = os.path.join(RESULTS, name)
    os.makedirs(out, exist_ok=True)

    np.random.seed(SEED)
    parameters = learn.initialize_parameters(784, hidden, 10)
… (98 linhas omitidas)
```

</details>

<details><summary>resultado</summary>

```
File created successfully at: /home/vinicius/ufmg/tp-2-intro-ia/experiments/run_experiments.py (file state is current in your context — no need to Read it back)
```

</details>

**Ferramenta: Bash** — Start experiment runner in background

```bash
nohup tp2-env/bin/python experiments/run_experiments.py > experiments/run.log 2>&1 &
echo "pid $!"
```

<details><summary>resultado</summary>

```
pid 66127
```

</details>

**Ferramenta: Bash** — Wait for learning-rate experiment to finish

```bash
timeout 590 bash -c 'until grep -q "melhor taxa\|Error\|Traceback" experiments/run.log; do sleep 10; done'; grep -v "Runtime\|frozen" experiments/run.log | grep -E "época (250|500)|==|->|Error|Trace"
```

<details><summary>resultado</summary>

```
== 1_taxa_aprendizado ==
   lr0.001_h64_e500 época 250: loss 2.3013 acc 0.1296 (59s)
   lr0.001_h64_e500 época 500: loss 2.2997 acc 0.1491 (118s)
   lr0.01_h64_e500 época 250: loss 2.2733 acc 0.3509 (59s)
   lr0.01_h64_e500 época 500: loss 2.0891 acc 0.4519 (118s)
   lr0.1_h64_e500 época 250: loss 0.4606 acc 0.8764 (61s)
   lr0.1_h64_e500 época 500: loss 0.3417 acc 0.9021 (110s)
   lr0.5_h64_e500 época 250: loss 0.2340 acc 0.9303 (52s)
   lr0.5_h64_e500 época 500: loss 0.1601 acc 0.9496 (99s)
-> melhor taxa: 0.5
== 2_neuronios ==
```

</details>

O experimento 1 terminou, e a melhor taxa foi **0,5**, com 94,96%. Com 0,001 a rede praticamente não saiu do lugar em 500 épocas. Agora rodam os de neurônios e de épocas.

**Ferramenta: Bash** — Wait for remaining experiments to finish

```bash
timeout 590 bash -c 'until grep -q "melhor configuração\|Error\|Traceback" experiments/run.log; do sleep 10; done'; grep -v "Runtime\|frozen" experiments/run.log | grep -E "época (500|50|100|200) |época 500:|época 50:|época 100:|época 200:|==|->|reaprov|Error|Trace" | sed -n '/2_neuronios/,$p'
```

<details><summary>resultado</summary>

```
== 2_neuronios ==
   lr0.5_h4_e500 época 50: loss 1.1273 acc 0.4411 (6s)
   lr0.5_h4_e500 época 100: loss 0.9171 acc 0.6544 (13s)
   lr0.5_h4_e500 época 200: loss 0.6996 acc 0.7722 (25s)
   lr0.5_h4_e500 época 500: loss 0.6776 acc 0.7883 (66s)
   lr0.5_h16_e500 época 50: loss 0.7312 acc 0.8061 (7s)
   lr0.5_h16_e500 época 100: loss 0.3834 acc 0.8870 (14s)
   lr0.5_h16_e500 época 200: loss 0.2895 acc 0.9151 (27s)
   lr0.5_h16_e500 época 500: loss 0.2102 acc 0.9372 (68s)
   (reaproveitando lr=0.5 h=64 e=500)
   lr0.5_h256_e500 época 50: loss 0.4720 acc 0.8705 (27s)
   lr0.5_h256_e500 época 100: loss 0.3290 acc 0.9067 (55s)
   lr0.5_h256_e500 época 200: loss 0.2463 acc 0.9286 (111s)
   lr0.5_h256_e500 época 500: loss 0.1462 acc 0.9544 (283s)
-> melhor nº de neurônios: 256
== 3_epocas ==
   lr0.5_h64_e50 época 50: loss 0.6849 acc 0.7884 (10s)
   lr0.5_h64_e100 época 50: loss 0.6849 acc 0.7884 (9s)
   lr0.5_h64_e100 época 100: loss 0.3380 acc 0.9024 (19s)
   lr0.5_h64_e200 época 50: loss 0.6849 acc 0.7884 (10s)
   lr0.5_h64_e200 época 100: loss 0.3380 acc 0.9024 (19s)
   lr0.5_h64_e200 época 200: loss 0.2585 acc 0.9243 (37s)
   (reaproveitando lr=0.5 h=64 e=500)
   (reaproveitando lr=0.5 h=256 e=500)
== melhor configuração: lr0.5_h256_e500 -> acc 0.9544 ==
```

</details>

**Ferramenta: Bash** — Show summary table and generated files

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia/experiments && column -s, -t results/summary.csv && cat results/melhor_configuracao.json && ls results/ && ls results/lr0.5_h256_e500/
```

<details><summary>resultado</summary>

```
experimento         treino            learning_rate  hidden_size  epochs  loss_final_treino  acuracia_final_teste  melhor_acuracia_teste  segundos  teve_nan
1_taxa_aprendizado  lr0.001_h64_e500  0.001          64           500     2.299724           0.149143              0.149143               117.7     False
1_taxa_aprendizado  lr0.01_h64_e500   0.01           64           500     2.089094           0.451857              0.451857               117.9     False
1_taxa_aprendizado  lr0.1_h64_e500    0.1            64           500     0.341717           0.902143              0.902143               110.1     False
1_taxa_aprendizado  lr0.5_h64_e500    0.5            64           500     0.160086           0.949643              0.949643               99.2      False
2_neuronios         lr0.5_h4_e500     0.5            4            500     0.677594           0.788286              0.836643               65.9      False
2_neuronios         lr0.5_h16_e500    0.5            16           500     0.210239           0.937214              0.937214               67.7      False
2_neuronios         lr0.5_h64_e500    0.5            64           500     0.160086           0.949643              0.949643               99.2      False
2_neuronios         lr0.5_h256_e500   0.5            256          500     0.146228           0.954429              0.954429               283.2     False
3_epocas            lr0.5_h64_e50     0.5            64           50      0.684857           0.788357              0.852429               10.0      False
3_epocas            lr0.5_h64_e100    0.5            64           100     0.337993           0.902429              0.904143               18.9      False
3_epocas            lr0.5_h64_e200    0.5            64           200     0.258546           0.924286              0.924286               37.5      False
3_epocas            lr0.5_h64_e500    0.5            64           500     0.160086           0.949643              0.949643               99.2      False
{
  "name": "lr0.5_h256_e500",
  "learning_rate": 0.5,
  "hidden_size": 256,
  "epochs": 500,
  "final_loss": 0.14622795587979698,
  "final_accuracy": 0.9544285714285714,
  "best_accuracy": 0.9544285714285714,
  "seconds": 283.2,
  "nan": false
}1_taxa_aprendizado_accuracies.png
1_taxa_aprendizado_losses.png
2_neuronios_accuracies.png
2_neuronios_losses.png
3_epocas_accuracies.png
3_epocas_losses.png
lr0.001_h64_e500
lr0.01_h64_e500
lr0.1_h64_e500
lr0.5_h16_e500
lr0.5_h256_e500
lr0.5_h4_e500
lr0.5_h64_e100
lr0.5_h64_e200
lr0.5_h64_e50
lr0.5_h64_e500
melhor_configuracao.json
… (11 linhas omitidas)
```

</details>

**Ferramenta: Read** — `/home/vinicius/ufmg/tp-2-intro-ia/experiments/results/2_neuronios_accuracies.png`

<details><summary>resultado</summary>

```
[imagem]
```

</details>

**Ferramenta: Read** — `/home/vinicius/ufmg/tp-2-intro-ia/experiments/results/1_taxa_aprendizado_losses.png`

<details><summary>resultado</summary>

```
[imagem]
```

</details>

Rodei os três experimentos e a melhor configuração. Nenhum treino teve `nan`, e a melhor configuração chegou a **95,44%** de acurácia de teste, acima dos 90% exigidos. Tudo levou ~17 min, contra os ~15 min que um único treino de 10.000 épocas levou na interface.

## Resultados

**1. Taxa de aprendizado** (64 neurônios, 500 épocas fixos)

| lr | Perda final (treino) | Acurácia (teste) | Tempo |
|---|---|---|---|
| 0,001 | 2,2997 | 14,9% | 118 s |
| 0,01 | 2,0891 | 45,2% | 118 s |
| 0,1 | 0,3417 | 90,2% | 110 s |
| **0,5** | **0,1601** | **95,0%** | 99 s |

**2. Neurônios na camada escondida** (lr 0,5 e 500 épocas fixos)

| Neurônios | Perda final | Acurácia | Tempo |
|---|---|---|---|
| 4 | 0,6776 | 78,8% (chegou a 83,7% no meio do treino) | 66 s |
| 16 | 0,2102 | 93,7% | 68 s |
| 64 | 0,1601 | 95,0% | 99 s |
| **256** | **0,1462** | **95,4%** | 283 s |

**3. Épocas** (lr 0,5 e 64 neurônios fixos)

| Épocas | Perda final | Acurácia | Tempo |
|---|---|---|---|
| 50 | 0,6849 | 78,8% (chegou a 85,2% no meio do treino) | 10 s |
| 100 | 0,3380 | 90,2% | 19 s |
| 200 | 0,2585 | 92,4% | 38 s |
| 500 | 0,1601 | 95,0% | 99 s |

**Melhor configuração:** lr 0,5, 256 neurônios e 500 épocas, com **95,44%** de acurácia.

## Observações para a discussão

- **Convergência × estabilidade.** Com 0,001 e 0,01, em 500 épocas a perda quase não sai de 2,30. Com 0,5, a rede converge rápido, mas a curva de perda tem **picos nas primeiras ~100 épocas**: passos grandes demais fazem a rede passar do ponto antes de se acomodar. Isso aparece no gráfico `1_taxa_aprendizado_losses.png`.
- **Capacidade.** Com 4 neurônios e lr 0,5, a acurácia **oscila entre 50% e 83%** até o fim do treino e termina abaixo do pico. A rede é pequena demais para essa taxa.
- **Capacidade × custo.** Passar de 64 para 256 neurônios rendeu só +0,5 ponto e custou 2,9 vezes mais tempo. Os retornos são decrescentes.
- **Os treinos de épocas são reproduzíveis.** Como a seed é fixa e o gradiente usa o lote completo, o treino de 50 épocas é idêntico às primeiras 50 épocas do de 500. Os quatro valores equivalem a ler pontos de uma mesma curva. Vale citar isso na metodologia.

## Arquivos

Tudo está em `/home/vinicius/ufmg/tp-2-intro-ia/experiments/`:

- `run_experiments.py`: o script, para citar na declaração de uso de IA.
- `results/summary.csv`: as tabelas acima.
- `results/1_taxa_aprendizado_*.png`, `2_neuronios_*.png` e `3_epocas_*.png`: gráficos comparativos de perda e acurácia por experimento.
- `results/<treino>/`: uma pasta por treino, com os pesos `.npy`, os CSVs no formato da interface (um ponto a cada 100 épocas), os CSVs época a época (`*_full.csv`) e as curvas `losses.png` e `accuracies.png`.
- `results/lr0.5_h256_e500/`: os pesos e as curvas da melhor configuração, que o relatório pede.

O treino de 50 épocas tem `losses.csv` e `accuracies.csv` **vazios**. É o mesmo comportamento da interface, que só registra a cada 100 épocas. Use os `_full.csv` desse treino.

## O que falta do seu lado

1. **Desenhar pelo menos 3 dígitos com o melhor modelo.** Para isso, os arquivos de `results/lr0.5_h256_e500/` precisam ir para `src/model/`. Ajuste o campo Neurônios para 256 e use o "Carregar Pesos". Atenção: copiar para lá sobrescreve o treino de 8 neurônios que recuperamos, então faça uma cópia antes. Se quiser, eu faço as duas coisas.
2. **Escrever o relatório** com essas tabelas e gráficos, declarando o uso de IA conforme a seção 6.

Se quiser tentar passar de 95,4%, dá para rodar a melhor configuração por mais épocas: 2.000 épocas com 256 neurônios levam uns 19 min. O PDF não exige isso.


---

## 👤 Aluno

Carregue o melhor modelo para o model para eu testar na interface.


---

## 🤖 Claude

**Ferramenta: Bash** — Check running server cwd and front-end load logic

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia && for p in $(pgrep -f "src/server.py\|server.py -port"); do echo "pid $p cwd=$(readlink /proc/$p/cwd) $(tr '\0' ' ' </proc/$p/cmdline)"; done; grep -n "function loadResults" -A25 src/static/sketch.js | grep -nE "hidden|NeuralNetwork|loadWeights|fetch"
```

<details><summary>resultado</summary>

```
6:156-    fetch(url)
11:161-        loadWeightsFromServer(data['parameters'])
```

</details>

**Ferramenta: Bash** — Check whether loading weights depends on hidden size field

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia/src/static && grep -n "setWeightsFromMatrix\|setBiasesFromMatrix\|initVisualization" -A8 neural.js | head -40; sed -n 84,100p sketch.js
```

<details><summary>resultado</summary>

```
179:    setWeightsFromMatrix(weights) {
180-      // the argument "weights" is an array that contains for each interlayer an object with the weights
181-      let weight;
182-  
183-      // go through interlayers
184-      for (let i = 0; i < this.weights.length; i++) {
185-  
186-        // go through nodes on the right side of interlayer i
187-        for (let j = 0; j < this.weights[i].data.length; j++) {
--
225:    setBiasesFromMatrix(biases) {
226-      // the argument "biases" is an array that contains for each interlayer an object with the biases
227-      let bias;
228-  
229-      // go through interlayers
230-      for (let i = 0; i < this.biases.length; i++) {
231-  
232-        // go through nodes on the right side of interlayer i
233-        for (let j = 0; j < this.biases[i].data.length; j++) {
--
246:    initVisualization(x, y, w, h, displayInputLayer) {
247-      this.visualize = true;
248-      this.x = x;
249-      this.y = y;
250-      this.h = h;
251-      this.w = w;
252-      this.displayInputLayer = displayInputLayer;
253-  
254-      if (this.displayInputLayer) {
    // Create notepad
    notepad = new Notepad(100, 130, 7, 28, 28);

    // Create neural network
    nn = new NeuralNetwork([784, hidden_size, 10]);
    nn.initVisualization(380, 30, 240, 400, false)

    // Create output elements
    marker = new Marker();
    outputDisplay = new OutputDisplay(width - 190, nn.y + nn.h / 2 - 230 / 2, 85, 230);

… (6 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Back up 8-neuron model and load best model into src/model

```bash
cd /home/vinicius/ufmg/tp-2-intro-ia && B=experiments/results/lr0.01_h8_e10000_interface && mkdir -p $B && cp -p src/model/{W1,b1,W2,b2}.npy src/model/losses.csv src/model/accuracies.csv src/losses.png src/accuracies.png $B/ && cp experiments/results/lr0.5_h256_e500/{W1,b1,W2,b2}.npy experiments/results/lr0.5_h256_e500/losses.csv experiments/results/lr0.5_h256_e500/accuracies.csv src/model/ && echo "== backup ($B) ==" && ls $B && echo "== src/model ==" && ls -la src/model && tp2-env/bin/python -c "
import sys; sys.path.insert(0,'src'); import numpy as np, learn
P = tuple(np.load(f'src/model/{n}.npy') for n in ['W1','b1','W2','b2'])
print('shapes:', [p.shape for p in P])
_, Xte, _, yte = learn.load_mnist()
print('acurácia de teste com src/model:', learn.accuracy(yte, learn.forward(Xte, P)[-1]))" 2>&1 | grep -v "Runtime\|frozen"
```

<details><summary>resultado</summary>

```
== backup (experiments/results/lr0.01_h8_e10000_interface) ==
W1.npy
W2.npy
accuracies.csv
accuracies.png
b1.npy
b2.npy
losses.csv
losses.png
== src/model ==
total 1628
drwxr-xr-x 2 vinicius vinicius    4096 Nov 17 20:15 .
drwxr-xr-x 6 vinicius vinicius    4096 Nov 17 20:15 ..
-rw-r--r-- 1 vinicius vinicius    6148 Nov 26  2025 .DS_Store
-rw------- 1 vinicius vinicius       0 Oct 28 17:28 Icon
-rw-r--r-- 1 vinicius vinicius 1605760 Nov 17 20:58 W1.npy
-rw-r--r-- 1 vinicius vinicius   20608 Nov 17 20:58 W2.npy
-rw-r--r-- 1 vinicius vinicius     125 Nov 17 20:58 accuracies.csv
-rw-r--r-- 1 vinicius vinicius    2176 Nov 17 20:58 b1.npy
-rw-r--r-- 1 vinicius vinicius     208 Nov 17 20:58 b2.npy
-rw-r--r-- 1 vinicius vinicius     125 Nov 17 20:58 losses.csv
shapes: [(784, 256), (1, 256), (256, 10), (1, 10)]
acurácia de teste com src/model: 0.9544285714285714
```

</details>

O melhor modelo (lr 0,5, 256 neurônios, 500 épocas) está em `src/model/`. Conferi: os shapes estão certos e a acurácia de teste lida desses arquivos é **95,44%**.

Antes de sobrescrever, salvei o treino de 8 neurônios em `experiments/results/lr0.01_h8_e10000_interface/`, com os pesos, os CSVs e os dois `.png`.

Para testar na interface:

1. **Suba o servidor a partir de `src`.** Não há nenhum servidor rodando agora.
   ```bash
   cd src && python server.py
   ```
2. Abra `http://localhost:5001` e coloque **256** no campo **Neurônios (camada escondida)** **antes** de carregar. Com outro valor, o front-end copia só uma parte dos pesos e a previsão sai errada.
3. Clique em **Carregar Pesos** e desenhe.

Desenhe grande e centralizado. O modelo continua sensível a deslocamento, como vimos com o seu "1".


---

## 👤 Aluno

E testei. O que acho que é interessante colocarmos no relatório é isso, que eu tive que aprender a desenhar o número deslocando o centro de massa para se aproximar aos desenhos do dataset. Se eu desenhar livremente, ele está com dificuldade extrema em entender alguns números como 8, 9 e similares. Veja se já temos tudo que precisamos para escrever o relatório. Se tivermos, crie o esqueleto dele baseado no relatório LaTeX que criamos para o TP1, que está na pasta ~/ufmg também. leia as diretrizes dele para lembrar se é o mesmo template que queremos utilizar. Se for, pode usar a mesma base dele, preencher o esqueleto. E depois iremos escrevendo o resto dele junto. Pode preencher de uma vez já o resultado dos experimentos. Esse ponto que comentamos do dataset, introdução, tudo mais.


---

## 🤖 Claude

**Ferramenta: Bash** — Find TP1 LaTeX report files in ~/ufmg

```bash
cd ~/ufmg && ls -la && find . -maxdepth 3 -not -path "*/tp-2-intro-ia/*" -not -path "*/node_modules*" -not -path "*/.git/*" \( -name "*.tex" -o -name "*.md" -o -name "*.sty" -o -name "*.bst" -o -name "*.bib" -o -name "*.pdf" \) | head -40
```

<details><summary>resultado</summary>

```
total 16
drwxr-xr-x  4 vinicius vinicius 4096 Nov 17 13:20 .
drwxr-x--- 38 vinicius vinicius 4096 Nov 17 20:16 ..
drwxr-xr-x  7 vinicius vinicius 4096 Oct 22 01:50 tp-1-intro-ia
drwxr-xr-x  6 vinicius vinicius 4096 Nov 17 20:39 tp-2-intro-ia
./tp-1-intro-ia/tp1-busca-competitiva.pdf
./tp-1-intro-ia/README.md
./tp-1-intro-ia/uso-de-ia/conversa-claude-code.md
./tp-1-intro-ia/reports/AnonymousSubmission2027.tex
./tp-1-intro-ia/reports/AnonymousSubmission2027.pdf
./tp-1-intro-ia/reports/relatorio.tex
./tp-1-intro-ia/reports/aaai2027.bib
./tp-1-intro-ia/reports/ReproducibilityChecklist.tex
./tp-1-intro-ia/reports/ReproducibilityChecklist.pdf
./tp-1-intro-ia/reports/aaai2027.sty
./tp-1-intro-ia/reports/CameraReady2027.pdf
./tp-1-intro-ia/reports/aaai2027.bst
./tp-1-intro-ia/reports/CameraReady2027.tex
Shell cwd was reset to /home/vinicius/ufmg/tp-2-intro-ia
```

</details>

**Ferramenta: Bash** — Read TP1 README, reports folder and AI-use notes

```bash
cd ~/ufmg/tp-1-intro-ia && cat README.md; ls -la reports; head -60 uso-de-ia/conversa-claude-code.md
```

**Ferramenta: Read** — `/home/vinicius/ufmg/tp-1-intro-ia/reports/relatorio.tex`

<details><summary>resultado</summary>

```
1	\documentclass[letterpaper]{article} % DO NOT CHANGE THIS
2	\usepackage{aaai2027}  % DO NOT CHANGE THIS
3	\usepackage[hyphens]{url}  % DO NOT CHANGE THIS
4	\usepackage{graphicx} % DO NOT CHANGE THIS
5	\urlstyle{rm} % DO NOT CHANGE THIS
6	\def\UrlFont{\rm}  % DO NOT CHANGE THIS
7	\usepackage{natbib}  % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
8	\usepackage{caption} % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
9	\frenchspacing  % DO NOT CHANGE THIS
10	\setlength{\pdfpagewidth}{8.5in} % DO NOT CHANGE THIS
11	\setlength{\pdfpageheight}{11in} % DO NOT CHANGE THIS
12	
13	\usepackage[brazil]{babel}   % "Tabela", "Figura" e hifenização em português
14	\usepackage{algorithm}
15	\usepackage{algorithmic}
16	\usepackage{booktabs}
17	
18	\pdfinfo{
19	/TemplateVersion (2027.2)
20	}
21	
22	\setcounter{secnumdepth}{0}
23	
24	\title{Busca Competitiva para Ligue-4: Minimax, Poda Alfa-Beta e Iterative Deepening}
25	\author{
26	    Vinicius Pinho Galvão
27	}
28	\affiliations{
29	    Departamento de Ciência da Computação\\
30	    Universidade Federal de Minas Gerais\\
31	    viniciuspgalvao@dcc.ufmg.br
32	}
33	
34	\begin{document}
35	
36	\maketitle
37	
38	\section{Introdução e Objetivo}
39	O Ligue-4 é jogado em um tabuleiro de 6 linhas por 7 colunas. Cada jogador solta uma peça por vez em uma coluna, a peça cai até a casa vazia mais baixa, e vence quem alinhar quatro peças na horizontal, na vertical ou na diagonal. É um jogo de dois jogadores, soma zero e informação perfeita, então se encaixa direto no modelo de busca competitiva.
40	
… (129 linhas omitidas)
```

</details>

<details><summary>resultado</summary>

```
# TP1 – Busca Competitiva (Ligue-4)

Agente de IA para **Ligue-4** (*Connect Four*), desenvolvido para a disciplina DCC642 – Introdução à Inteligência Artificial (UFMG, 2026/2). O agente evolui do **Minimax** com profundidade limitada para a **poda Alfa-Beta** e, por fim, para o **Iterative Deepening** com limite de tempo, todos guiados por uma função heurística de avaliação.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `src/search.py` | O agente: Minimax, Alfa-Beta, Iterative Deepening e heurística. Ponto de entrada: `choose_move(board, player, config)`. |
| `src/server.py` | Servidor Flask que serve a interface web (p5.js) e chama o agente. |
| `src/experiments.py` | Script que roda partidas automáticas (sem servidor) e gera as tabelas do relatório. |
| `src/resultados.log` | Saída bruta da execução dos experimentos. |
| `reports/` | Relatório no template AAAI-27. |
| `diretrizes.md` | Enunciado do trabalho. |

## Como executar

ˋˋˋbash
python -m venv tp1-env
source tp1-env/bin/activate
pip install -r src/requirements.txt

cd src
python server.py            # interface em http://localhost:5001
ˋˋˋ

Para reproduzir os experimentos:

ˋˋˋbash
cd src
python experiments.py                          # exp. 1, 2 e 3, 10 partidas por confronto
python experiments.py --exp 1 2 -n 20          # escolhe experimentos e número de partidas
python experiments.py --exp 3 --csv out.csv    # salva os resultados agregados em CSV
ˋˋˋ

Opções úteis: `--opening` (jogadas aleatórias de abertura nos exp. 2 e 3, padrão 2), `--ab-depth` (profundidade do Alfa-Beta no exp. 3, padrão 5) e `--seed` (padrão 42).

## O agente

**Convenção de pontuação.** O Jogador 1 é sempre o maximizador e o Jogador 2 o minimizador. Estados terminais valem `±1.000.000` (descontada a profundidade, para preferir vitórias mais rápidas e derrotas mais tardias); empate vale 0.
… (138 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Collect hardware, versions, LaTeX tools and TP1 figures

```bash
lscpu | grep -E "Model name|^CPU\(s\)"; tp2-env/bin/python -c "import sys, numpy, sklearn; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| sklearn', sklearn.__version__)"; which pdflatex latexmk bibtex 2>&1; ls ~/ufmg/tp-1-intro-ia/reports/Figures; grep -c "bibliography\|cite" ~/ufmg/tp-1-intro-ia/reports/relatorio.tex
```

**Ferramenta: Bash** — Confusion matrix and shift sensitivity of best model

```bash
tp2-env/bin/python - <<'EOF' 2>&1 | grep -v "Runtime\|frozen"
import sys; sys.path.insert(0, "src")
import numpy as np, learn
P = tuple(np.load(f"experiments/results/lr0.5_h256_e500/{n}.npy") for n in ["W1","b1","W2","b2"])
Xtr, Xte, ytr, yte = learn.load_mnist()
yh = learn.forward(Xte, P)[-1]; t = yte.argmax(1); p = yh.argmax(1)
cm = np.zeros((10,10), int); np.add.at(cm, (t, p), 1)
np.savetxt("experiments/results/lr0.5_h256_e500/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
print("acerto por dígito:", "  ".join(f"{i}:{cm[i,i]/cm[i].sum():.1%}" for i in range(10)))
off = sorted([(cm[i,j], i, j) for i in range(10) for j in range(10) if i != j], reverse=True)[:6]
print("maiores confusões:", ", ".join(f"{i}->{j}: {n}" for n, i, j in off))
imgs = Xte.reshape(-1, 28, 28)
rows = []
for d in range(5):
    r = learn.accuracy(yte, learn.forward(np.roll(imgs, d, axis=2).reshape(-1, 784), P)[-1])
    rows.append(r)
print("deslocamento 0-4px:", "  ".join(f"{d}px:{r:.1%}" for d, r in enumerate(rows)))
# diagonal (direita+baixo) 2px, como no desenho (≈2px acima/esquerda)
r = learn.accuracy(yte, learn.forward(np.roll(np.roll(imgs, 2, axis=2), 2, axis=1).reshape(-1, 784), P)[-1])
print(f"2px direita + 2px baixo: {r:.1%}")
EOF
```

<details><summary>resultado</summary>

```
CPU(s):                               12
Model name:                           AMD Ryzen 5 3600 6-Core Processor
python 3.14.10 | numpy 2.3.5 | sklearn 1.7.2
figure1.pdf
figure2.pdf
0
```

</details>

<details><summary>resultado</summary>

```
acerto por dígito: 0:97.5%  1:97.8%  2:94.9%  3:94.0%  4:94.8%  5:94.2%  6:97.5%  7:96.6%  8:93.2%  9:93.5%
maiores confusões: 4->9: 37, 9->4: 32, 5->3: 28, 8->3: 26, 3->5: 23, 9->7: 20
deslocamento 0-4px: 0px:95.4%  1px:90.6%  2px:71.3%  3px:48.2%  4px:26.1%
2px direita + 2px baixo: 39.6%
```

</details>

**Ferramenta: Bash** — Check best model on the reconstructed drawing

```bash
tp2-env/bin/python -c "
import sys; sys.path.insert(0, '/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad')
import drawing
from drawing import *
drawing.P = tuple(np.load(f'experiments/results/lr0.5_h256_e500/{n}.npy') for n in ['W1','b1','W2','b2'])
for dx in range(5): show(f'melhor modelo: desenho +{dx}px', np.roll(x_draw, dx, axis=1))
show('melhor modelo: padrão MNIST', mnist_style(x_draw))
np.save('/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/drawing_1.npy', x_draw)
" 2>&1 | grep -v "Runtime\|frozen"
```

<details><summary>resultado</summary>

```
melhor modelo: desenho +0px            previsão=8  softmax top3: 8:52.4%, 5:42.1%, 6:3.2%
melhor modelo: desenho +1px            previsão=1  softmax top3: 1:72.3%, 8:20.0%, 5:6.0%
melhor modelo: desenho +2px            previsão=1  softmax top3: 1:99.3%, 9:0.2%, 8:0.2%
melhor modelo: desenho +3px            previsão=1  softmax top3: 1:64.5%, 9:22.5%, 7:7.2%
melhor modelo: desenho +4px            previsão=7  softmax top3: 7:32.4%, 3:32.0%, 9:29.6%
melhor modelo: padrão MNIST            previsão=1  softmax top3: 1:95.5%, 9:1.4%, 4:1.2%
```

</details>

**Ferramenta: Bash** — Set up reports folder with AAAI style files

```bash
cp /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/drawing_1.npy experiments/results/desenho_1_canvas.npy && mkdir -p reports/figures && cp ~/ufmg/tp-1-intro-ia/reports/aaai2027.sty ~/ufmg/tp-1-intro-ia/reports/aaai2027.bst reports/ && ls reports
```

**Ferramenta: Write** — `/home/vinicius/ufmg/tp-2-intro-ia/experiments/make_figures.py`

<details><summary>conteúdo</summary>

```
"""Gera as figuras do relatório (reports/figures/) a partir de experiments/results/.

Uso: python experiments/make_figures.py   (a partir da raiz do projeto, depois de run_experiments.py)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import learn  # noqa: E402

RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
W = 3.3  # largura de uma coluna AAAI, em polegadas


def full(name, metric):
    return np.loadtxt(os.path.join(RES, name, f"{metric}_full.csv"), delimiter=",")


# Figura 1: perda por taxa de aprendizado
plt.figure(figsize=(W, 2.0))
for lr in [0.001, 0.01, 0.1, 0.5]:
    plt.plot(np.arange(1, 501), full(f"lr{lr}_h64_e500", "losses"), label=f"$\\alpha$ = {lr}")
plt.xlabel("Época"); plt.ylabel("Perda (treino)")
plt.grid(True, linestyle="--", alpha=.6); plt.legend()
plt.savefig(os.path.join(FIG, "fig_taxa.pdf")); plt.close()

# Figura 2: acurácia por número de neurônios
plt.figure(figsize=(W, 2.0))
for h in [4, 16, 64, 256]:
    plt.plot(np.arange(1, 501), full(f"lr0.5_h{h}_e500", "accuracies"), label=f"{h} neurônios")
plt.xlabel("Época"); plt.ylabel("Acurácia (teste)")
plt.grid(True, linestyle="--", alpha=.6); plt.legend(loc="lower right")
plt.savefig(os.path.join(FIG, "fig_neuronios.pdf")); plt.close()

# Figura 3: curvas da melhor configuração
fig, ax = plt.subplots(1, 2, figsize=(W, 1.6))
ax[0].plot(np.arange(1, 501), full("lr0.5_h256_e500", "losses"), color="red")
ax[0].set_xlabel("Época"); ax[0].set_ylabel("Perda (treino)")
ax[1].plot(np.arange(1, 501), full("lr0.5_h256_e500", "accuracies"), color="blue")
ax[1].set_xlabel("Época"); ax[1].set_ylabel("Acurácia (teste)")
for a in ax:
    a.grid(True, linestyle="--", alpha=.6)
fig.tight_layout()
plt.savefig(os.path.join(FIG, "fig_melhor.pdf")); plt.close()

# Figura 4: o "1" desenhado na interface, original e deslocado, e a média dos "1" do MNIST
P = tuple(np.load(os.path.join(RES, "lr0.5_h256_e500", f"{n}.npy")) for n in ["W1", "b1", "W2", "b2"])
draw = np.load(os.path.join(RES, "desenho_1_canvas.npy"))
… (22 linhas omitidas)
```

</details>

<details><summary>resultado</summary>

```
aaai2027.bst
aaai2027.sty
figures
```

</details>

<details><summary>resultado</summary>

```
File created successfully at: /home/vinicius/ufmg/tp-2-intro-ia/experiments/make_figures.py (file state is current in your context — no need to Read it back)
```

</details>

**Ferramenta: Bash** — Generate report figures

```bash
tp2-env/bin/python experiments/make_figures.py 2>&1 | grep -v "Runtime\|frozen" && cd reports/figures && for f in *.pdf; do tp2-env/bin/python -c "" 2>/dev/null; done; which pdftoppm magick convert 2>/dev/null
```

<details><summary>resultado</summary>

```
Exit code 1
figuras em /home/vinicius/ufmg/tp-2-intro-ia/reports/figures : ['fig_deslocamento.pdf', 'fig_melhor.pdf', 'fig_neuronios.pdf', 'fig_taxa.pdf']
/usr/bin/convert
```

</details>

**Ferramenta: Read** — `/home/vinicius/ufmg/tp-2-intro-ia/reports/figures/fig_deslocamento.pdf`

<details><summary>resultado</summary>

```
PDF file read: /home/vinicius/ufmg/tp-2-intro-ia/reports/figures/fig_deslocamento.pdf (16KB)
[documento PDF]
```

</details>

**Ferramenta: Read** — `/home/vinicius/ufmg/tp-2-intro-ia/reports/figures/fig_melhor.pdf`

<details><summary>resultado</summary>

```
PDF file read: /home/vinicius/ufmg/tp-2-intro-ia/reports/figures/fig_melhor.pdf (13KB)
[documento PDF]
```

</details>

**Ferramenta: Edit** — `/home/vinicius/ufmg/tp-2-intro-ia/experiments/make_figures.py`

Antes:

```
panels = [(draw, "Desenho original", label(draw)),
          (np.roll(draw, 2, axis=1), "Deslocado 2 px", label(np.roll(draw, 2, axis=1))),
          (mean1 / mean1.max(), "Média dos \"1\" (MNIST)", "")]
```

Depois:

```
panels = [(draw, "(a) desenho", label(draw)),
          (np.roll(draw, 2, axis=1), "(b) +2 px à direita", label(np.roll(draw, 2, axis=1))),
          (mean1 / mean1.max(), "(c) média dos \"1\"", "MNIST (teste)")]
```

**Ferramenta: Bash** — Regenerate report figures

```bash
tp2-env/bin/python experiments/make_figures.py 2>&1 | grep -v "Runtime\|frozen"
```

**Ferramenta: Read** — `/home/vinicius/ufmg/tp-2-intro-ia/reports/figures/fig_deslocamento.pdf`

<details><summary>resultado</summary>

```
The file /home/vinicius/ufmg/tp-2-intro-ia/experiments/make_figures.py has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>

<details><summary>resultado</summary>

```
figuras em /home/vinicius/ufmg/tp-2-intro-ia/reports/figures : ['fig_deslocamento.pdf', 'fig_melhor.pdf', 'fig_neuronios.pdf', 'fig_taxa.pdf']
```

</details>

<details><summary>resultado</summary>

```
PDF file read: /home/vinicius/ufmg/tp-2-intro-ia/reports/figures/fig_deslocamento.pdf (16.5KB)
[documento PDF]
```

</details>

**Ferramenta: Bash** — Compute shift sensitivity for 8, 64, 256-neuron models

```bash
tp2-env/bin/python - <<'EOF' 2>&1 | grep -v "Runtime\|frozen"
import sys; sys.path.insert(0, "src")
import numpy as np, learn
_, Xte, _, yte = learn.load_mnist(); imgs = Xte.reshape(-1, 28, 28)
for name in ["lr0.01_h8_e10000_interface", "lr0.5_h64_e500", "lr0.5_h256_e500"]:
    P = tuple(np.load(f"experiments/results/{name}/{n}.npy") for n in ["W1","b1","W2","b2"])
    accs = [learn.accuracy(yte, learn.forward(np.roll(imgs, d, axis=2).reshape(-1, 784), P)[-1]) for d in range(5)]
    print(name, " ".join(f"{a*100:.1f}" for a in accs))
EOF
```

<details><summary>resultado</summary>

```
lr0.01_h8_e10000_interface 90.6 84.9 64.0 41.8 21.7
lr0.5_h64_e500 95.0 89.9 71.3 50.2 29.1
lr0.5_h256_e500 95.4 90.6 71.3 48.2 26.1
```

</details>

**Ferramenta: Write** — `/home/vinicius/ufmg/tp-2-intro-ia/reports/relatorio.tex`

<details><summary>conteúdo</summary>

```
\documentclass[letterpaper]{article} % DO NOT CHANGE THIS
\usepackage{aaai2027}  % DO NOT CHANGE THIS
\usepackage[hyphens]{url}  % DO NOT CHANGE THIS
\usepackage{graphicx} % DO NOT CHANGE THIS
\urlstyle{rm} % DO NOT CHANGE THIS
\def\UrlFont{\rm}  % DO NOT CHANGE THIS
\usepackage{natbib}  % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\usepackage{caption} % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\frenchspacing  % DO NOT CHANGE THIS
\setlength{\pdfpagewidth}{8.5in} % DO NOT CHANGE THIS
\setlength{\pdfpageheight}{11in} % DO NOT CHANGE THIS

\usepackage[brazil]{babel}   % "Tabela", "Figura" e hifenização em português
\usepackage{booktabs}
\usepackage{amsmath}

\pdfinfo{
/TemplateVersion (2027.2)
}

\setcounter{secnumdepth}{0}

% Marcador das partes a escrever. Apagar todas as ocorrências antes de entregar
% (grep -n nota relatorio.tex).
\newcommand{\nota}[1]{\textit{[#1]}}

\title{Redes Neurais Artificiais para Classificação de Dígitos Manuscritos (MNIST)}
\author{
    Vinicius Pinho Galvão
}
\affiliations{
    Departamento de Ciência da Computação\\
    Universidade Federal de Minas Gerais\\
    viniciuspgalvao@dcc.ufmg.br
}

\begin{document}

\maketitle

% ---------------------------------------------------------------------------
\section{Introdução e Objetivo}
\nota{A escrever. Pontos:
(1) o problema: classificar imagens 28$\times$28 de dígitos 0--9 do MNIST (70.000 imagens), classificação multiclasse com $C = 10$;
(2) o que foi feito: uma RNA de duas camadas (uma escondida, uma de saída) implementada do zero em NumPy, treinada por gradiente descendente;
(3) o objetivo: entender cada peça (propagação, ativações, perda, gradiente) e medir o efeito da taxa de aprendizado, do número de épocas e do número de neurônios;
(4) o resultado principal: 95,44\% de acurácia de teste com $\alpha = 0{,}5$, 256 neurônios e 500 épocas; e a observação de que, na interface, foi preciso aprender a desenhar com o centro de massa próximo ao do MNIST.}

% ---------------------------------------------------------------------------
\section{Metodologia}

\subsection{Dados}
\nota{A escrever. Fatos: \texttt{load\_mnist} carrega o MNIST do OpenML; pixels normalizados para $[0,1]$ (divisão por 255); divisão aleatória 80/20 (\texttt{random\_state} = 42), 56.000 imagens de treino e 14.000 de teste; rótulos em \emph{one-hot}. Cada imagem vira um vetor de 784 entradas, e as matrizes têm um exemplo por \textbf{linha}.}

\subsection{Arquitetura e Inicialização}
\nota{A escrever. Fatos: 784 entradas, $H$ neurônios escondidos (hiperparâmetro), 10 saídas. $W_1$ ($784 \times H$) e $W_2$ ($H \times 10$) com \texttt{np.random.randn} $\times\, 0{,}01$; $b_1$ ($1 \times H$) e $b_2$ ($1 \times 10$) zerados. Por que pesos pequenos e aleatórios: quebrar a simetria entre neurônios e começar perto da região linear das ativações.}

\subsection{Propagação e Funções de Ativação}
\begin{align*}
Z_1 &= X W_1 + b_1, & A_1 &= \mathrm{ReLU}(Z_1),\\
… (156 linhas omitidas)
```

</details>

<details><summary>resultado</summary>

```
File created successfully at: /home/vinicius/ufmg/tp-2-intro-ia/reports/relatorio.tex (file state is current in your context — no need to Read it back)
```

</details>

Montei o esqueleto do relatório em `reports/relatorio.tex`, com os mesmos arquivos de estilo do TP1 (`aaai2027.sty` e `.bst`). Não consegui compilar: o WSL não tem `pdflatex`. Então ainda não sei quantas páginas ele ocupa.

**Um ponto sobre a seção 6 do PDF.** Ela proíbe gerar o relatório "na íntegra" com IA. Por isso preenchi os **dados**, mas não escrevi o **texto corrido**. Fiz como no TP1, em que a sua declaração dizia que o Claude "montou o esqueleto".

**O que já está pronto:**
- A estrutura que o enunciado pede: Introdução e Objetivo, Metodologia, Experimentos e Resultados, Discussão, Conclusão e Declaração de Uso de IA.
- A **Tabela 1** com os 12 treinos dos três experimentos (perda final, acurácia e tempo). Conferi cada número com o `summary.csv`.
- A **Tabela 2**, com a acurácia no teste deslocado de 0 a 4 px para os modelos de 8, 64 e 256 neurônios.
- As fórmulas de propagação e da perda.
- As figuras, todas em `reports/figures/`:
  - perda por taxa de aprendizado;
  - acurácia por número de neurônios;
  - curvas da melhor configuração;
  - o seu "1" original (o modelo de 256 responde **8**), deslocado 2 px (responde **1, com 99%**) e a média dos "1" do MNIST.

**O que fica com você:** cada seção tem um bloco `\nota{...}` com os pontos e números a cobrir, inclusive o do centro de massa que você levantou. Você transforma em texto, e eu reviso, como fizemos com o código. Antes de entregar, apague todos com `grep -n nota reports/relatorio.tex`.

**Respondendo se temos tudo:** quase. Os dados estão completos. Faltam três coisas:
1. **As suas observações com pelo menos 3 dígitos desenhados com o melhor modelo**, que o enunciado pede explicitamente. Já tenho os pontos que você citou (8 e 9 difíceis, ter que ajustar o centro de massa) e os números que os confirmam: 8 e 9 são justamente os piores dígitos no teste, com 93,2% e 93,5%. Se quiser incluir prints, isso pesa no limite de 3 páginas.
2. **O link para as transcrições** na declaração de uso de IA, como você fez no TP1.
3. **Conferir o limite de 3 páginas.** São 6 floats (2 tabelas e 4 figuras). Se passar, a Figura 2 (neurônios) é a primeira candidata a sair, porque a tabela já cobre o essencial.

Também ficaram no projeto o `experiments/make_figures.py` e o desenho reconstruído do print, em `experiments/results/desenho_1_canvas.npy`, para as figuras poderem ser regeneradas.


---

## 👤 Aluno

Salve essa conversa que tivemos no repositório, como fizemos no tp1 e link o arquivo da conversa com base no GitHub. https://github.com/Vipigal/mnist-neural-network-training


---

## 🤖 Claude

**Ferramenta: Bash** — Inspect TP1 conversation export format

```bash
cd ~/ufmg/tp-1-intro-ia && ls -la uso-de-ia/ && find . -maxdepth 3 -iname "*export*" -o -maxdepth 3 -iname "*conversa*" | grep -v node_modules; grep -n "^## \|^### " uso-de-ia/conversa-claude-code.md | head -20; grep -c "" uso-de-ia/conversa-claude-code.md; sed -n 60,140p uso-de-ia/conversa-claude-code.md
```

**Ferramenta: Bash** — Check git state and session log files

```bash
git status --short | head -20; git branch -a; git remote -v; git log --oneline | head -5; ls -la ~/.claude/projects/-home-vinicius-ufmg-tp-2-intro-ia/ | head
```

<details><summary>resultado</summary>

```
M reports/relatorio.tex
* main
  remotes/origin/main
origin	https://github.com/Vipigal/mnist-neural-network-training.git (fetch)
origin	https://github.com/Vipigal/mnist-neural-network-training.git (push)
3a53b69 add initial files
total 13312
drwxr-xr-x  4 vinicius vinicius    4096 Nov 17 22:08 .
drwx------ 15 vinicius vinicius    4096 Nov 17 13:22 ..
drwxr-xr-x  4 vinicius vinicius    4096 Nov 17 22:08 81074f1a-6a8f-4377-b602-d0b59fdaa41c
-rw-------  1 vinicius vinicius 4007268 Nov 17 22:09 81074f1a-6a8f-4377-b602-d0b59fdaa41c.jsonl
-rw-------  1 vinicius vinicius 3807800 Nov 17 13:27 910e3cb8-5992-4d9d-a920-68854485e7e2.jsonl
-rw-------  1 vinicius vinicius 5289710 Nov 17 22:10 a8f413a2-5b8a-4a1e-9b9b-ef7912c81857.jsonl
-rw-------  1 vinicius vinicius  492921 Nov 17 13:23 bc0c5913-4a96-459c-b06a-f86c847e810c.jsonl
drwxr-xr-x  2 vinicius vinicius    4096 Nov 17 13:22 memory
```

</details>

<details><summary>resultado</summary>

```
total 196
drwxr-xr-x 2 vinicius vinicius   4096 Oct 22 01:48 .
drwxr-xr-x 7 vinicius vinicius   4096 Oct 22 01:50 ..
-rw-r--r-- 1 vinicius vinicius 189622 Oct 22 00:43 conversa-claude-code.md
./uso-de-ia/conversa-claude-code.md
12:## 👤 Aluno
216:## 🤖 Claude
453:## Bugs encontrados
483:## O que esperar depois de corrigir
492:## 👤 Aluno
501:## 🤖 Claude
590:## O bug: a vez não passa para o adversário na raiz
613:## Por que sempre a coluna 0
622:## Mais dois detalhes
630:## 👤 Aluno
637:## 🤖 Claude
643:## Ordem sugerida dos próximos passos
652:## A ideia da Alfa-Beta
663:## O que muda no seu `minimax`
673:## Como saber se está certo
682:## Por que a ordenação importa tanto
695:## 👤 Aluno
702:## 🤖 Claude
706:## Regra de ouro: um único ponto de vista
722:## Componente 1: valorização do centro
4354
Testando move 0 no minimax... Player = 2 | Depth = 5
Limite de recursao do minimax atingido! Player = 2 | Depth = 6
Best score para move 0 = inf Player = 2 | Depth = 5
Testando move 1 no minimax... Player = 2 | Depth = 5
Limite de recursao do minimax atingido! Player = 2 | Depth = 6
Best score para move 1 = inf Player = 2 | Depth = 5
Testando move 2 no minimax... Player = 2 | Depth = 5
Limite de recursao do minimax atingido! Player = 2 | Depth = 6
Best score para move 2 = inf Player = 2 | Depth = 5
Testando move 3 no minimax... Player = 2 | Depth = 5
Limite de recursao do minimax atingido! Player = 2 | Depth = 6
Best score para move 3 = inf Player = 2 | Depth = 5
Testando move 4 no minimax... Player = 2 | Depth = 5
Limite de recursao do minimax atingido! Player = 2 | Depth = 6
… (68 linhas omitidas)
```

</details>

**Ferramenta: Bash** — See how TP1 export showed tool calls

```bash
cd ~/ufmg/tp-1-intro-ia && sed -n 200,260p uso-de-ia/conversa-claude-code.md; grep -nE "^> |🔧|<details|Ferramenta|\*\*Bash\*\*|`Bash`" uso-de-ia/conversa-claude-code.md | head -15
```

**Ferramenta: Bash** — Inspect session JSONL structure

```bash
cd ~/.claude/projects/-home-vinicius-ufmg-tp-2-intro-ia && python3 - <<'EOF'
import json, collections
f = "a8f413a2-5b8a-4a1e-9b9b-ef7912c81857.jsonl"
types = collections.Counter(); first = {}
with open(f) as fh:
    for line in fh:
        o = json.loads(line); t = o.get("type"); types[t] += 1
        if t not in first: first[t] = {k: (str(v)[:150]) for k, v in o.items()}
print(types)
for t, o in first.items(): print("==", t, json.dumps(o, ensure_ascii=False)[:700])
EOF
```

<details><summary>resultado</summary>

```
Limite de recursao do minimax atingido! Player = 2 | Depth = 6
Best score para move 1 = inf Player = 2 | Depth = 5
Testando move 2 no minimax... Player = 2 | Depth = 5
Best score para move 2 = -1 Player = 2 | Depth = 5
Testando move 3 no minimax... Player = 2 | Depth = 5
Limite de recursao do minimax atingido! Player = 2 | Depth = 6
Best score para move 3 = -1 Player = 2 | Depth = 5
Testando move 4 no minimax... Player = 2 | Depth = 5
Limite de recursao do minimax atingido! Player = 2 | Depth = 6
Best score para move 4 = -1 Player = 2 | Depth = 5
Testando move 5 no minimax... Player = 2 | Depth = 5
ˋˋˋ


---

## 🤖 Claude

**Ferramenta: Bash** — List project files

ˋˋˋbash
ls -la && find . -name "*.py" -not -path "./.venv/*" | head -50
ˋˋˋ

<details><summary>resultado</summary>

ˋˋˋ
total 332
drwxr-xr-x 5 vinicius vinicius   4096 Oct 20 22:40 .
drwxr-xr-x 3 vinicius vinicius   4096 Oct 20 21:36 ..
drwxr-xr-x 7 vinicius vinicius   4096 Oct 20 22:40 .git
-rw-r--r-- 1 vinicius vinicius      9 Oct 20 22:40 .gitignore
drwxr-xr-x 5 vinicius vinicius   4096 Oct 20 21:36 src
-rwxr-xr-x 1 vinicius vinicius 311881 Oct 20 21:36 tp1-busca-competitiva.pdf
drwxr-xr-x 5 vinicius vinicius   4096 Oct 20 21:38 tp1-env
./src/search.py
./src/server.py
./tp1-env/lib/python3.10/site-packages/blinker/__init__.py
./tp1-env/lib/python3.10/site-packages/blinker/base.py
./tp1-env/lib/python3.10/site-packages/blinker/_utilities.py
… (38 linhas omitidas)
```

</details>

_(saída omitida: dados internos do log da sessão)_

**Ferramenta: Write** — `/tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/export_conversa.py`

<details><summary>conteúdo</summary>

```
"""Exporta a sessão do Claude Code (JSONL) para Markdown, no formato usado no TP1."""
import json
import re
import sys

SRC, DST = sys.argv[1], sys.argv[2]
MAX_LINES = 40

HEADER = """# Conversa com o Claude Code (uso de IA no TP2)

Transcrição da sessão do Claude Code (modelo Claude Opus 5.5) usada durante o TP2 de Introdução à Inteligência Artificial (DCC642, UFMG).

- Exportada automaticamente do log local da sessão; o raciocínio interno do modelo não é incluído.
- Chamadas de ferramentas (comandos executados, arquivos lidos/editados) aparecem resumidas; saídas longas foram truncadas.
- Mensagens do usuário estão sem edição.

"""


def clean_user_text(t):
    t = re.sub(r"<system-reminder>.*?</system-reminder>", "", t, flags=re.S)
    t = re.sub(r"<pasted_content[^>]*>", "\nˋˋˋtext\n", t)
    t = re.sub(r"</pasted_content[^>]*>", "\nˋˋˋ\n", t)
    return t.strip()


def trunc(text, n=MAX_LINES):
    lines = text.rstrip().split("\n")
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[:n]) + f"\n… ({len(lines) - n} linhas omitidas)"


def fence(text, lang=""):
    text = text.replace("ˋˋˋ", "ˋˋˋ")
    return f"ˋˋˋ{lang}\n{text}\nˋˋˋ"


def tool_use(block):
    name, inp = block["name"], block.get("input", {})
    if name == "Bash":
        head = f"**Ferramenta: Bash** — {inp.get('description', '')}".rstrip(" —")
        return f"{head}\n\n{fence(trunc(inp.get('command', '')), 'bash')}"
    if name == "Read":
        return f"**Ferramenta: Read** — `{inp.get('file_path', '')}`"
    if name == "Write":
        return (f"**Ferramenta: Write** — `{inp.get('file_path', '')}`\n\n"
                f"<details><summary>conteúdo</summary>\n\n{fence(trunc(inp.get('content', ''), 60))}\n\n</details>")
    if name == "Edit":
        return (f"**Ferramenta: Edit** — `{inp.get('file_path', '')}`\n\n"
                f"Antes:\n\n{fence(trunc(inp.get('old_string', '')))}\n\nDepois:\n\n{fence(trunc(inp.get('new_string', '')))}")
    if name == "Skill":
        args = inp.get("args", "")
        return f"**Skill: {inp.get('skill', '')}**" + (f" — {args}" if args else "")
    return f"**Ferramenta: {name}**\n\n{fence(trunc(json.dumps(inp, ensure_ascii=False, indent=1)), 'json')}"


def tool_result(block):
    c = block.get("content", "")
    if isinstance(c, list):
… (67 linhas omitidas)
```

</details>

<details><summary>resultado</summary>

```
File created successfully at: /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/export_conversa.py (file state is current in your context — no need to Read it back)
```

</details>

**Ferramenta: Bash** — Export session to markdown and list user turns

```bash
mkdir -p uso-de-ia && python3 /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/export_conversa.py ~/.claude/projects/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857.jsonl uso-de-ia/conversa-claude-code.md && wc -lc uso-de-ia/conversa-claude-code.md && grep -c "^## 👤 Aluno" uso-de-ia/conversa-claude-code.md && grep -n "^## 👤 Aluno" -A3 uso-de-ia/conversa-claude-code.md | grep -v "^--$" | cut -c1-160 | head -80
```

<details><summary>resultado</summary>

```
ok uso-de-ia/conversa-claude-code.md
  4461 202571 uso-de-ia/conversa-claude-code.md
19
13:## 👤 Aluno
14-
15-Estou implementando o passo a passo como descrito no pdf em docs. Avalie minha funcao initialize_parameters para ver se ficou correta e me ajude a implementa
16-
189:## 👤 Aluno
190-
191-O que o Z representa aqui?
192-
259:## 👤 Aluno
260-
261-Ok, agora sobre o softmax, me explique como ela funciona e como posso implementa-la em numpy.
262-
381:## 👤 Aluno
382-
383-implementei a relu e softmax, roda os testes do pdf
384-
512:## 👤 Aluno
513-
514-Teste agora.
515-
670:## 👤 Aluno
671-
672-E agora? Se funcionar, me explique o que os parametros axis fazem, porque a normalizacao pelo max de Z funciona e como ela funciona. Prefira exemplos visuai
673-
853:## 👤 Aluno
854-
855-Para o cross_entropy_loss, o que eu utilizaria como valor de C e m? Pra mim C é 10 pois é o numero de classes que iremos utilizar no mnist (output_size do
856-
950:## 👤 Aluno
951-
952-Implementei o cross_entropy_loss da maneira que soube fazer. Verifique sua corretude e se estiver ok, me ensine a fazer com np.sum
953-
1138:## 👤 Aluno
1139-
1140-Meu language server do python no vscode está crashando constantemente enquanto estou escrevendo código. Recebo alguns erros como 
1141-
1640:## 👤 Aluno
… (39 linhas omitidas)
```

</details>

**Ferramenta: Bash** — Check export for sensitive data and completeness

```bash
F=uso-de-ia/conversa-claude-code.md; for p in "gmail" "system-reminder" "accountUuid" "organizationUuid" "Base directory" "CLAUDE.md" "skill \`" "imagem anexada" "Stopping the server timed out" "api_key\|token=" ; do printf "%-35s %s\n" "$p" "$(grep -c -- "$p" $F)"; done; sed -n 1,40p $F; grep -n "skill \`" $F | head
```

<details><summary>resultado</summary>

```
gmail                               0
system-reminder                     3
accountUuid                         1
organizationUuid                    1
Base directory                      0
CLAUDE.md                           0
skill `                             2
imagem anexada                      1
Stopping the server timed out       8
api_key\|token=                     0
# Conversa com o Claude Code (uso de IA no TP2)

Transcrição da sessão do Claude Code (modelo Claude Opus 5.5) usada durante o TP2 de Introdução à Inteligência Artificial (DCC642, UFMG).

- Exportada automaticamente do log local da sessão; o raciocínio interno do modelo não é incluído.
- Chamadas de ferramentas (comandos executados, arquivos lidos/editados) aparecem resumidas; saídas longas foram truncadas.
- Mensagens do usuário estão sem edição.



---

## 👤 Aluno

Estou implementando o passo a passo como descrito no pdf em docs. Avalie minha funcao initialize_parameters para ver se ficou correta e me ajude a implementar a RELU. Eu sei como a relu funciona, porém nao sei manipular arrays do numpy o suficiente para tal. Em pseudocodigo eu faria algo como: para cada i, j de Z, eu rodaria um max Z[i][j], 0 e salvaria o resultado na matriz, devolvendo ela.


---

## 🤖 Claude

**Ferramenta: Bash** — List project files and docs

ˋˋˋbash
ls -la && ls docs && find . -name "*.py" -not -path "./.venv/*" -not -path "*/site-packages/*" | head -20
ˋˋˋ

<details><summary>resultado</summary>

ˋˋˋ
… (12 linhas omitidas)
```

</details>

A conversa foi exportada (19 mensagens suas). Estou terminando de limpar o arquivo: tirando os identificadores da conta, que apareceram numa saída de comando.

**Ferramenta: Bash** — Locate remaining sensitive markers in export

```bash
F=uso-de-ia/conversa-claude-code.md; grep -n "system-reminder\|accountUuid\|organizationUuid" $F | cut -c1-200
```

<details><summary>resultado</summary>

```
4365:== session-owner {"type": "session-owner", "accountUuid": "[omitido]", "organizationUuid": "[omitido]", "sessionId": "a8f413a2-5b8a-4a1e-9b9b
4369:== attachment {"parentUuid": "1a854070-a57d-43fb-88c7-4e8bde874a66", "isSidechain": "False", "attachment": "{'type': 'environment', 'snapshot': {'workingDirectory': '/home/vinicius/ufmg/tp-2-intr
4371:== api-request-blob {"type": "api-request-blob", "sessionId": "a8f413a2-5b8a-4a1e-9b9b-ef7912c81857", "hash": "01fa3736c49a6ce15d5a2a2ebaea86396efd9d88eb0f8ddabfc8b067aa14430e", "message": "{'rol
4410:    t = re.sub(r"", "", t, flags=re.S)
```

</details>

**Ferramenta: Bash** — Omit internal log dump and re-export conversation

```bash
cd /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad && python3 - <<'EOF'
p = "export_conversa.py"
s = open(p).read()
s = s.replace('''                if b.get("type") == "tool_result":
                    r = tool_result(b)''', '''                if b.get("type") == "tool_result":
                    if b.get("tool_use_id") in OMIT:
                        out.append("_(saída omitida: dados internos do log da sessão)_\\n")
                        continue
                    r = tool_result(b)''')
s = s.replace('''                elif b.get("type") == "tool_use":
                    section("assistant")''', '''                elif b.get("type") == "tool_use":
                    if "JSONL" in b.get("input", {}).get("description", ""):
                        OMIT.add(b["id"])
                    section("assistant")''')
s = s.replace("out, last_role = [HEADER], None", "out, last_role = [HEADER], None\nOMIT = set()")
open(p, "w").write(s)
EOF
cd /home/vinicius/ufmg/tp-2-intro-ia && python3 /tmp/claude-1000/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857/scratchpad/export_conversa.py ~/.claude/projects/-home-vinicius-ufmg-tp-2-intro-ia/a8f413a2-5b8a-4a1e-9b9b-ef7912c81857.jsonl uso-de-ia/conversa-claude-code.md && grep -c "accountUuid\|organizationUuid\|api-request-blob" uso-de-ia/conversa-claude-code.md; grep -n "saída omitida" uso-de-ia/conversa-claude-code.md
```

<details><summary>resultado</summary>

```
ok uso-de-ia/conversa-claude-code.md
6
4359:_(saída omitida: dados internos do log da sessão)_
```

</details>

**Ferramenta: Bash** — Find remaining matches of internal identifiers

```bash
grep -n "accountUuid\|organizationUuid\|api-request-blob" uso-de-ia/conversa-claude-code.md | cut -c1-160
```

<details><summary>resultado</summary>

```
4496:F=uso-de-ia/conversa-claude-code.md; for p in "gmail" "system-reminder" "accountUuid" "organizationUuid" "Base directory" "CLAUDE.md" "skill \`" "imagem an
4504:accountUuid                         1
4505:organizationUuid                    1
4552:F=uso-de-ia/conversa-claude-code.md; grep -n "system-reminder\|accountUuid\|organizationUuid" $F | cut -c1-200
4558:4365:== session-owner {"type": "session-owner", "accountUuid": "[omitido]", "organizationUuid": "24afa780-88b3-4fa0-9f16-9a50f4d
4560:4371:== api-request-blob {"type": "api-request-blob", "sessionId": "a8f413a2-5b8a-4a1e-9b9b-ef7912c81857", "hash": "01fa3736c49a6ce15d5a2a2ebaea86396efd9d8
```

</details>
