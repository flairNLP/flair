import unittest
import torch

from flair.data import Sentence
from unittest.mock import patch, MagicMock, call, mock_open

from flair.embeddings import (
    GazetteerEmbeddings,
)


class GazetteerEmbeddingsTest(unittest.TestCase):
    full_match_hash_dict = [
        {'ORG': {'042ab0dfc27eaa2f099847e67b6e5d2aad0632ca43915a4044bd1679afdea9c4': 'Jekabpils '
                                                                                     'Agrobusiness '
                                                                                     'College',
                 '0756ae1986d2db1435f86f5dc3c88c2b248c69e9a0ee47b738f50ccfa14db43b': '+EU',
                 '0757ebc975de34615ecb2fa8a72332bae77de92f02d2ec11cbb71252a3dbae35': '1 '
                                                                                     'C.D.Gramsci '
                                                                                     '- '
                                                                                     'S.M. '
                                                                                     'Pende',
                 '1abd994913a8e127729f77467d079b9f90f0ba5036ef51377c3974a995655e94': 'Дом-музей '
                                                                                     'Островского',
                 '1bf8da82cd2a790c703c9d218d8456240bb6053a50a1b1279ab5f08d472a4e59': 'Художньо-меморіальний '
                                                                                     'музей '
                                                                                     'Олекси '
                                                                                     'Новаківського',
                 '1eb7dae4733bd462ee787ecda616c942af2a6f7a45f4b7deafcb8021d55e6ed5': '1 '
                                                                                     'C.D. '
                                                                                     'Bovio '
                                                                                     'Ruvo',
                 '20a938f7d79470dbe7b752a49c8afe6440c145288743253afdf5d44cf421be76': 'โรงเรียนหาดใหญ่รัฐประชาสรรค์',
                 '2367635407641f9e45cf657b181e857a8284558aa8a4050b0a8f75929c7e9921': '1 '
                                                                                     '2 '
                                                                                     '3 '
                                                                                     'Stella',
                 '2672cff63819f5699860aa4801f572d8f1d95f68bedad10788ba4dce2e8d592a': '1 '
                                                                                     '- '
                                                                                     'Antonio '
                                                                                     'Salvetti '
                                                                                     'Colle '
                                                                                     'V.E.',
                 '2770b5e44b91b1a08d80932112065fdba5fb6b914fa25da28971b3ccb57c1399': '"Mons. '
                                                                                     'Aurelio '
                                                                                     'Sorrentino" '
                                                                                     'Diocesan '
                                                                                     'Museum',
                 '3787d2fc52ded133990c2222413458de5093b480898a246a3e4dae4195daca4e': '"Pedro '
                                                                                     'Arrupe" '
                                                                                     'Political '
                                                                                     'Training '
                                                                                     'Institute',
                 '39d1126b63b3d8c94afae19eb6bff6b2bc2479a30aeaaea0494b7fd84557108e': "'48 "
                                                                                     'Smallholders '
                                                                                     'Party',
                 '3cb87ec228b9f45f509f5ea6653ce3b8ca79fd690b7a7fb32c290c7dc9662deb': 'Русскій '
                                                                                     'BЂстникъ',
                 '3db42605fbecf1bc958dd3b49214dab0b212f98faa6c2f554003010edf2fc371': '!Xunkhwesa '
                                                                                     'Combined '
                                                                                     'School',
                 '4180abcabf7205d4eebd3207b4f3d3a64996ea67249afd7d0567f13d3d0dda79': '#Minas '
                                                                                     'para '
                                                                                     'Todos',
                 '43c883ed864a1f341bdc0871c6d6cc94cc540b3e76810a245d861f120f63aca9': ' '
                                                                                     'National '
                                                                                     'Archives '
                                                                                     'and '
                                                                                     'Library '
                                                                                     'Agency',
                 '48572d90c63e8058358ddb4f00268783fd4c81fbc81e09c7c85e9c25947be2fa': '#Nome?',
                 '4c541b9f6806f222d85f808a32a31c91c7c39731c34add983c7f95f3e6c73b8f': "' "
                                                                                     'Cogozzo '
                                                                                     '- '
                                                                                     'Don '
                                                                                     'Eugenio '
                                                                                     'Mazzi '
                                                                                     "'",
                 '506daaf36bfba905fd3b9a1e4812e2f337928f51bee1b93759817348834701f6': '""Onderzoek '
                                                                                     '- '
                                                                                     'en '
                                                                                     'Documentatiecentrum '
                                                                                     'Beweging '
                                                                                     'van '
                                                                                     'Mensen '
                                                                                     'met '
                                                                                     'Laag '
                                                                                     'Inkomen '
                                                                                     'en '
                                                                                     'Kinderen"',
                 '5c438e97c114c78c01f7c7c67545045bb1353cf63332e3f7a936e7e5a77a31a4': 'Ąžuolas',
                 '612424833745b97d86760e766eeaf5ca8fca3887433975bf7b636bbd8e467ac3': 'Land '
                                                                                     'Tenure '
                                                                                     'Reform '
                                                                                     'Association',
                 '62d013a04a30895503c9ab7647747010c53d3da96982e942126929bb54a1e65c': '1 '
                                                                                     '- '
                                                                                     'I.C. '
                                                                                     'Nocera '
                                                                                     'Inferiore',
                 '6ff25b9536d5242e1d4f685cee13e818fb314375a772d6817acaefe4661021c0': 'Sandys',
                 '7dc491fe5b0d2c420a7926f237524b7f66f325ce9327943b20b14fa80d3b1722': 'Raymond '
                                                                                     'and '
                                                                                     'Beverly '
                                                                                     'Sackler '
                                                                                     'Institute '
                                                                                     'for '
                                                                                     'Biological, '
                                                                                     'Physical '
                                                                                     'and '
                                                                                     'Engineering '
                                                                                     'Sciences, '
                                                                                     'Yale '
                                                                                     'University',
                 '86da81aa263a42fcc93d946c881a4d6a1f071c6a43df218533406353a60d14c3': 'Ειδικό '
                                                                                     'Γυμνάσιο '
                                                                                     '- '
                                                                                     'Ειδικό '
                                                                                     'ΓΕΛ '
                                                                                     'Αθηνών',
                 '8da7cf09ee2ca8980fe7344db10fdca23c039fdd58d96f91cac68f99315a1b56': "'t "
                                                                                     'Vliegend '
                                                                                     'Peert '
                                                                                     'Centre '
                                                                                     'for '
                                                                                     'Old '
                                                                                     'Arts',
                 '91bd8858d2e91f763043d7cefc9a5cd01a64074dfee9e8101b16b584510e7e3b': 'Sandys '
                                                                                     'Fort '
                                                                                     'Spring',
                 '9859410a534a38b200bf2833ce4a0b1bd3dd7d72e402a27ac2220f5569b07859': '!Bang!',
                 'ab1dfe7c8b3c74a161ab57456ea64745c9bbc22e2cf40336f4ebc2ffcfc0629d': "'A. "
                                                                                     "Rosmini' "
                                                                                     '- '
                                                                                     'Robecchetto '
                                                                                     'C/I',
                 'addb5744f32ae7d122bb31388930b322c27600744ae2422c3181ac1bbbf756f5': '알로이시오중학교',
                 'c553b1f9f4f635e2024ddd7b75a0a302f4395fbafb081b4eddc286a08b96c318': "''A. "
                                                                                     "Ilvento''-Grassano",
                 'c71810bac396ce7ca7ce543394bbb68e8ad09c2b60508db08839c7e603405e8a': 'ΑΕΝ '
                                                                                     'Dromolaxia',
                 'cb89fddbd5552609a04ad3825f4641e43992d507290da317b83c998cbfe830a5': '"Octavian '
                                                                                     'Goga" '
                                                                                     'Cluj '
                                                                                     'County '
                                                                                     'Library',
                 'ce8b1cbf0cf8a55142a1057b599e1e5280f33ebe69bffa60b786c8fb6d7fbcfc': '"Red '
                                                                                     'Terror" '
                                                                                     "Martyrs' "
                                                                                     'Memorial '
                                                                                     'Museum',
                 'e084dc8b18dd3dc2af05e9648f22455a79275b02ef168fdefa0965c1b0b45aa1': ', '
                                                                                     'Berhampur',
                 'e4b32307915591b9adffd8ecd355b443afe9e2d2243bc7f68ef06ba66f02c156': 'หนังสือพิมพ์ผู้จัดการ',
                 'ebca0aa23aaad7181cc90eea623035ef012401670eb26a4818a0942d9cee5253': "' "
                                                                                     'Villa '
                                                                                     'Saviola '
                                                                                     "'",
                 'f54e5285e5d126fc7f9105465c7b05706609ff0d93235b45476eb2a72a6044ff': 'Партия '
                                                                                     'Развития '
                                                                                     'Украины',
                 'fddb18f27dbef55eda112453a0d5cb79de83193298a5c70a195c1838e5e86ab7': 'America '
                                                                                     'de '
                                                                                     'Manta'}},
        {'LOC': {'0c8c3a2fa93844da231ca95bbcf83b87043f12018cf9ce7d43941dfac003d4ee': '"La '
                                                                                     'Cetățui" '
                                                                                     'archaeological '
                                                                                     'site '
                                                                                     'in '
                                                                                     'Tismana',
                 '1044503959ca6f68c28dd430c113f86c88b56437ab6771e2fbe000fcee457d71': 'дерево '
                                                                                     'спарапетов '
                                                                                     'в '
                                                                                     'Иджеване',
                 '176d6d00160fb0fdef78b5bc97ccaa22527030c92e145f2b8a400017c9d6dfdf': "'Aklé "
                                                                                     'Aoukâr',
                 '179d1bcaa7c7d4d959937771c4a8417daf5f38224ccb6cc0bba010464a1c7f8d': 'تاهات '
                                                                                     'أتاكور',
                 '23da455b57dee764dcaa7291531c902d4fc48c27bdb296371eeb7af36d85dc80': 'Sandøydrætten',
                 '2d9a153c127000433c78f0c2aab35f88028e571b23b61cec5250935dfbddc4e1': "'Adam "
                                                                                     'Dihé',
                 '2f94eda89f2d4247df0ed0c165e1634ff5d262a34c3ec41e660110a9b337f823': "'Aklé "
                                                                                     "'Âouâna",
                 '4c662dc6874a68bb21b5fa87fa55e3defd2aa7fc19da8d231d882374fde95fe3': 'آبشار '
                                                                                     'شوی',
                 '53f0bc99de1de5625d767b3c12ed74e3236868108ce903c93a36f9c42a629d5d': 'კუნძული '
                                                                                     'ორპირი',
                 '63e0018101b2f98a3fc7dd8061da7c60f2e0f81cfc25c49f9c183625d7d87e1a': 'Sandøygrunnen',
                 '640824fa320606b36f54a69bf6c6e512e8cba3e9b9696dbee20bfccf10803f01': '"La '
                                                                                     'Grande '
                                                                                     'Cité"',
                 '838131df5d837cfa397cf075ee25d60711a027284ee075c07fc2bba9b645ce7f': 'Ķūķu '
                                                                                     'cliffs',
                 '91bd8858d2e91f763043d7cefc9a5cd01a64074dfee9e8101b16b584510e7e3b': 'Sandys '
                                                                                     'Fort '
                                                                                     'Spring',
                 'a075685aed4594e82cef30bab032e640ec7c0a0d9278b3f188a49c2c5626e7ce': "' "
                                                                                     'Azeïlât '
                                                                                     'Aḥmoûd',
                 'acf34d688070655acfda254d81c45c0241a5231cc9ddfc3487cb6c49c136bafc': "'Achar "
                                                                                     'Teïdoûm',
                 'c7934c706819e0c9ab2255951895059489816ae14dd2e2651ba1f59fb4c0126f': 'Опорски '
                                                                                     'рид',
                 'd5b74cb5345b0f1eff09d45fd88ce97e1c47562da6c53808bfcada6310637e97': "'Aklé "
                                                                                     'el '
                                                                                     'Liyene',
                 'd69e77def744c4d88cfb6e9744ef8f39776a735c80eb44181220393f5ec07ee1': 'Sandøyfjorden',
                 'd956b380efa9e586584f45e77b7f58f85afd3cd0ca1f1bc714514e13d69ceee1': "'Aleb "
                                                                                     'Bel '
                                                                                     "'Aïd",
                 'e0cee2bb883a8f30fa0178607619780e674a977e2f2b401786d21694c88ad395': "'Adade "
                                                                                     'Yus '
                                                                                     'Mountain',
                 'e4d1f6c90a489880727bf72b32a3edacd9a6db8bfae8c9cd36877cbf81292ef8': '"Congarees" '
                                                                                     'Site',
                 'e618c6c8b8e4d12fe171476a3f6d3bbef6b75cdcf958805a2f78e5a0cecee20d': '"El '
                                                                                     'Salto" '
                                                                                     'Tonatico',
                 'ee27a9c62e5585d70dbf4c8a1bb75a3c47415775db9168094b120b17332c4af4': 'Голоух'}}]

    partial_match_hash_dict = [
        {'B-ORG': {'158091db4440664561964d12f92789afedd863a0273a4dbd7271e856c348c137': "''A.",
                   '1f0ca711df81520887afe0dca099652a249e7eda60348be7327d432b02298652': 'Antonio',
                   '2618749c92c9d603fe8c7eb0cc51770e0763a8e4a85bc1992238a5cde7649feb': '#Minas',
                   '41539615a60bd98176aa79cb83739c0b89181f0587eb0ea23097ab1c26abdd6e': 'I.C.',
                   '41c39ae3a0ebdb5400d6676c8bf743cf9f496d0f5208ad6fb9543e391298d9b9': 'Cogozzo',
                   '446a366896cca834e0906c37f6caf85e44227627f7aa6d260f48b2f35231b335': '!Xunkhwesa',
                   '49ef3ad156bcc99adad2ecabe043c5cccee3433b7afbb24422a2da5dda8c8a78': '"Octavian',
                   '4c0e998a40def661697913757cdee6925ab5084a8ff3200c6c2727c0e9191052': 'C.D.',
                   '4c1111dc2143bb4281374e4d7e0b285931d0ebeeefcc495b04f717beb19e9ecc': 'ΑΕΝ',
                   '5830b7b4c5bab8a5e2f556e9282e7daf7df40688acf8a20d95c0a6527cc08761': 'Партия',
                   '5e8a78d8eb5afc0c04660ff2f775ef781953e966df253ff59bc70d1126c5a9be': '"Red',
                   '5fab0a07631207178d0bcbaf7f89d6520489aaf5f2b0b26e0f94755192963d44': 'Художньо-меморіальний',
                   '5fca53dde09b7b13682a0acdb36bf496b01e4a3805cbf04ad5d65848bf1f6d17': 'Ειδικό',
                   '61d9655f09fc49a91111424ea0fd7f5fcd6e6ccc0a7e6b655af8acb4cbaa3c30': 'Русскій',
                   '6ff25b9536d5242e1d4f685cee13e818fb314375a772d6817acaefe4661021c0': 'Sandys',
                   '865b30e2fae6d778db9b748c30705b8ea7ec051a61182126f110d484c4347055': 'National',
                   '87abb5cfe01d88a48190ab1e0fc78ef285963632988aaf48aee2b2559b154f29': "'A.",
                   '98d7f113aa13e67287bf6e9657d690be9c89226c9dbd50453897022484ad2771': 'America',
                   'a11d1f9c25b075acd321f71356c39100076c1210550aafda94ddf831bd58351c': "'t",
                   'a903d05a0c465202e8d5d2dc72f37a7be694e2c22f13c3a1b5df763d1a056d47': '"Mons.',
                   'ac335678c3e9f3c2811a89cbb32eb51227675aaeb033d4202e66a7020f20f6ba': '""Onderzoek',
                   'aef7c74ca0863bf2a53186aacf7dcc6a676f1f2f2b636711dcea5ecd0d85fc4b': 'Jekabpils',
                   'afad4c579e50189cd1eb668530ae3c2ba8f9b99a6a39972e0c4bd4e5df8bf51e': 'Villa',
                   'b6baff9358ddd8f760a539c3f4b36a3314876e86597f3465a3623b6be3f3cf29': 'Land',
                   'bcccf6342db59411da83bf1e5ffd0d13ba666c46ca3b36181014ee8d55797b1e': 'C.D.Gramsci',
                   'bdca723ca6aaac12cd0f47f0407974977027a88847d20f5013f7d4000c483d2f': "'48",
                   'f2745f6867b663da3b7a320edbd94a6301facf3671f594e0442f7fc7cf1fabd7': 'Дом-музей',
                   'faa016a27649a937f39168b6b9d5455f5abe8c83b3a410b335b05b3643c8696d': '"Pedro',
                   'ffb595919dd6a431bc74948317cd56be39802b6a2c9a9f0d08606c7b01edb250': 'Raymond'},
         'E-ORG': {'05f00f61bbd7cbb7f47de945257784f53623dd1e2d130ae87e7f123631863ecf': 'College',
                   '067dea0e353ec6a73cd8215b23bc00f5da78a72739c75e1406078992be7e1940': 'Institute',
                   '0a2df2f79ffdd6ec9dfef402a2c8f4b746e57e21b7f60bd493d1d65b3e1c2e82': 'Новаківського',
                   '24f9c03b653ef3a3d3841d61e2654bdebecd2534aec3f55c0a43f64d77a018ed': 'Saviola',
                   '2da78dba9e51cce36c87030dea6993568cd15921355c4ae7d505deb42ed6cbf6': 'V.E.',
                   '477ce0ff961d8e73955ead9d8f76b99b9cf78a74fcbbc5e21fd21867c9ac5d55': 'C/I',
                   '56f57ca00dd8ccc9b0f53462611d516260d14bb61ebbde03decb73b384a227d4': 'Украины',
                   '58c448e7fbeb01e450541e43ae7ee04206459649ce481328e2c5b105d9a65e60': 'Dromolaxia',
                   '69ea90a1cbfab427471ee2513d37d9b77def06bb1a15a09eb13a316b23e66a55': 'Mazzi',
                   '6cd2e331855da821ec18452ef337985a5dc2f2ee6eb0bc2b0c894768d5a95c32': 'School',
                   '86f357cfb8c2c1002c1228b554b8d00808cc17f85ed35451fcf3aa0403ce94d2': 'Agency',
                   '8f4d8ea9d5682625f7195bdb8f7a99a2bbda08c3718cac3e73d9ab3b540f367c': 'Kinderen"',
                   '9d16dbea1d6c9fd624f2b32bcfbb5f8c41766bd7182a79d017dfeb02e4ac00bf': 'Museum',
                   '9d38443e2220ded5f60ba39b85acdad3af96f810e359bb741f6d273470c7bf9b': 'University',
                   'a61d9e176cfacdc4b316a8d6131e5b5e87cb645e40a00fde853e5bdb54d42496': 'BЂстникъ',
                   'b7de525cb64841e6d9e048d9b14d6ba89107848b5f60d09bff8ef915346e573d': 'Островского',
                   'bd02b9a7d71dac67a21b2758f11975bc491cc834c22edd9fa598fed97405aaa9': 'Todos',
                   'c1d4342085d23e14f4acf13172f864adcc495d0f6ebf3470d55d499ee6721136': 'Association',
                   'c397787f729aa0c81a5688d95004ead744c9f4d838cd82f2c6327c5e929d4056': 'Manta',
                   'd6ed1a97ca510a4f458db208fc8f5902e345a490f199ad664da35af8708566a2': "Ilvento''-Grassano",
                   'dac618a536e3a9a7ae7e6071d47ecf5d4c79f55276e9f7c5a3889b8daba19fef': 'Arts',
                   'db29e82a9287fc2834d8596af212c2da01149ebab1f0a48437b1f0c990af7647': 'Spring',
                   'dc20b3d5d2cddf82fad332821ca5e9a4efdcede4a273aa193d2b495bdcc92825': 'Library',
                   'dfa82f21b3f3709c94f525d2d63cb6b2955e5a5d2430757cbedfa230f116b2ed': 'Party',
                   'e0421dea6a02430a7a6d13989b0aae00b0b02bf2a7014e9533592a0777de4ab6': 'Ruvo',
                   'eafff18d6f989fa2fc7aecc8ce6eeaf5e55fecdd3c92ff7458fc757b786f97c8': 'Αθηνών',
                   'ee51fb7de5c6976ee6cdfe1ca52d9148dfdd0ed81017a4a691ea1fcf94ce90ce': 'Pende',
                   'fa89db5de050f6e897ee34e69a1b2d2ed5de9cef07c1e6c4732bf1450053741f': 'Inferiore'},
         'I-ORG': {'067dea0e353ec6a73cd8215b23bc00f5da78a72739c75e1406078992be7e1940': 'Institute',
                   '0d4914d04308ecd1b4485995b8490af7aaabae62c7162efcba9deb6bdb60c907': 'Agrobusiness',
                   '10c22bcf4c768b515be4e94bcafc71bf3e8fb5f70b2584bcc8c7533217f2e7f9': 'for',
                   '136c87dec4695cefef030fd6dc7c32faeb6280a795fda70b017197b60e3b7873': 'County',
                   '172b727276ca44c302c2c708c55f51991c5b36e40445fad59718e10331ceba46': 'Goga"',
                   '24691e68058219ff6d9fcb12d4d00eda0473420efe7d8a273cc9803bd04299ef': 'ΓΕΛ',
                   '246feaa0bc431c7066d7ecf87f053ddc45354e4028c7d3cf0421e65d64b5b04e': 'Colle',
                   '24b6bef25e6e1efd34fdef52e772b39bfd818a77b5eea88f1badfe790361b7e5': 'Don',
                   '27ff2e6793e118b64ea6091ba1a919c8c0a349b48f53ec7e13b7204826915d02': 'Peert',
                   '286d4b99281e7330fd5d63334c2f176881eb42ccd35b9d90bcc65949c703563b': 'Cluj',
                   '2ab14761d20d110e3f31acad8ba5b9ae18a78f4d8180222c3323aa5525d663ed': 'Γυμνάσιο',
                   '36a798e3f39239ccdec697ee348d2ce0a7b8d2d9087940496e2ca5d0853399f2': 'Training',
                   '45c19322f7b9337e148397c84067d4f15cb76b6a1226acb467a6c5a2b1584b70': 'Centre',
                   '55de6ac58e9321f89dc36e793025c0f56ed6a5d6df2a46842155f2bada8b4d0b': 'Laag',
                   '6201111b83a0cb5b0922cb37cc442b9a40e24e3b1ce100a4bb204f4c63fd2ac0': 'and',
                   '6b8565df6822cecfbadd14cec5e70284f06866727a714b3fe11f2906b9355aa2': 'Smallholders',
                   '6e6a4ab082c008cf9c70ec5d9a833ea75d40af57eca50c4a4401d21e18913cc3': 'Sackler',
                   '6efcbfb1f9f195ced75feb43f202ab14a8cd87832cef086976b16658e23944a8': 'met',
                   '718c673574346d499e8dcafc8bb9caab685a4484d6ef41eb3cdf03ca9cc8db2b': 'Beweging',
                   '729bb48d0f86a9e240a9839bf8c50f62f7ed973338514432666cf53c5d30c9ca': 'Engineering',
                   '7b7ae728e701311b6713511c139b8759a66b0b68c9f4592e0f7eb590df3b20ce': 'Bovio',
                   '7c80a715e3297855ad4bb01a5ecb572e144266714350cabfa967db34cb9ff15b': 'Robecchetto',
                   '8036ed3754e114f252cd6ee41c696ff52c7fe8bd912f11a52c53aedf9772db58': 'Combined',
                   '8401bb4fc11fab60b1c5d3fe883de4eff7eb2b64b73436d00002744fec18a9c6': 'Documentatiecentrum',
                   '8407c1496d06c9c32030e0b0faa327d57f6644610f83b4e6e01c1388d4d8fff2': 'Mensen',
                   '872d94afc42c307f5834a1c80f2af285220398cde599df3985c29f2ec6aa9e98': 'Tenure',
                   '89266e5c0273292e0699c6636b0a0079ae74d28acd1220a5acef38884ae02b53': 'Развития',
                   '8f5ce74d9ad696475a7f3ea58b0940fd327bcc0d23fa2ddf4ce1cde3b53c2b4f': 'Sciences,',
                   '92ebaa7ab6de626bded0fcfa0251cc5f5fd7073752053fc9ac096df17a0cd6f4': 'Eugenio',
                   '959a45d44e6fcf58361ed004681556fe50129f2109e817dec098c00c9e5d2578': 'de',
                   '968a0feb5a106d55b6aea300268353bc20ddfcb54d6eecbb0eafc533741c55cd': 'Biological,',
                   '970b1897af0abab6127d77e3306c1a712c5d931a2719ecd338f57411e534ec3c': 'van',
                   '98f9a7f76b88aea417deb4aa6ace485090c15ddd9b0dc7a698be03d062590d37': 'Beverly',
                   '9a62e43b1b31ce5f7d39398b5dafe97db73d4ae4ec20e98fe2798bef4f915b12': "Rosmini'",
                   '9ee0706ab6ff51d38e98f69a3846c3a2a9ecc8142ac9d91bbfd65934f8a1e000': 'S.M.',
                   'a1453f380fa9f1a08e84d4703e9c168fda1fb9a36976c41a03c8af842aa04ce5': 'para',
                   'a89383f1a4302676318d7ff58fcce2c8bf4fd46ac48aa55fcbebb6a340ca2c60': 'Physical',
                   'b47e4baec4153d4e1c083049ad048ea408642331f9f1003a07112cdd7a3beaf2': 'Arrupe"',
                   'b69feea9efa8615cd491957b7ef828efb410c2dd676ca063759a689af4fef9b1': 'Fort',
                   'bc261a00ef7fe70749eae118d7054671d9776e87261dd3eb61fedecd3590ae50': 'Political',
                   'bca97160f4e1211fe659338d0a9705a7dff8aa3ea2e1be1cc1958100a33962c2': 'Old',
                   'c10504dce02cf0ef592e6babfb1a37864bfd17c35dcae4edb1afd868482a0b2b': 'Олекси',
                   'c256f404a5f8108aa5cf5d1450c63717a64c0395d4ecd01769c2cb88f516b19c': 'Aurelio',
                   'c31bc980f6615e28c8c49509d2ecb434852af9e85882e70d21ae3eba26429eb2': 'Yale',
                   'c59b7a87ac4ab51ccadf8bc941f530bfebcda9614145cffaa7534f01c7fedf25': 'Reform',
                   'c61e432cc6e9d6849c7af8023267bd4f9e9a5f9232395165c9d41686d8f3090a': 'Salvetti',
                   'c74faf0ab9c7081f828cdfd2b429a38b25448cc8fe67b9aaf71186a133752250': 'Terror"',
                   'c92f3fff59eec71e5d68425703a1112617496191da76b263e514183ccc40d2a1': 'Memorial',
                   'd5d1974081ea3658335fa831c5d8f4976f1a44d6ded69511dc69e6b0c2090615': 'Inkomen',
                   'dbd3a49d0d906b4ed9216b73330d2fb080ef2f758c12f3885068222e5e17151c': 'en',
                   'dc20b3d5d2cddf82fad332821ca5e9a4efdcede4a273aa193d2b495bdcc92825': 'Library',
                   'de768e1fa4095ba6395efd171021322f981544c4c02e20bbc9c2985ad1f23a6b': 'Sorrentino"',
                   'e143c07fd74064bddbe9e0887e47e825cc0c9860bbd272bd30cacf5cf58ac6b7': 'музей',
                   'e19e879f21f11b110d06381cdcde4c428d49f113015dce8f763d05a20081ab3f': 'Diocesan',
                   'e2005d5422f1718ded2ec088c5112f59d72bfddc2615eed82192cc66d2b99ce6': 'Nocera',
                   'e404aa80d8df4adb314c5c0e324a018068ee22660abde94e0dcc644bef866b31': 'Archives',
                   'f791743cb18491be7ded2a602cac14d53de5cca00789afcbd4c7d6ae8358f198': "Martyrs'",
                   'fc727bf1c90a9f7c1a16f20c25189fa1e62e62fdf4151f2c6dcb423c89ee4cb4': 'Vliegend'},
         'S-ORG': {'0756ae1986d2db1435f86f5dc3c88c2b248c69e9a0ee47b738f50ccfa14db43b': '+EU',
                   '20a938f7d79470dbe7b752a49c8afe6440c145288743253afdf5d44cf421be76': 'โรงเรียนหาดใหญ่รัฐประชาสรรค์',
                   '48572d90c63e8058358ddb4f00268783fd4c81fbc81e09c7c85e9c25947be2fa': '#Nome?',
                   '5c438e97c114c78c01f7c7c67545045bb1353cf63332e3f7a936e7e5a77a31a4': 'Ąžuolas',
                   '6ff25b9536d5242e1d4f685cee13e818fb314375a772d6817acaefe4661021c0': 'Sandys',
                   '9859410a534a38b200bf2833ce4a0b1bd3dd7d72e402a27ac2220f5569b07859': '!Bang!',
                   'a26f87e3b4e4a4a1479b09e0fc28f14b39f55dfb4a18dc08283e9a7ce909ff19': 'Berhampur',
                   'addb5744f32ae7d122bb31388930b322c27600744ae2422c3181ac1bbbf756f5': '알로이시오중학교',
                   'e1df3d7704cb8c50945b3655ae79ec58bcf79702c55877a0d2c9625fb68b2b49': 'Stella',
                   'e4b32307915591b9adffd8ecd355b443afe9e2d2243bc7f68ef06ba66f02c156': 'หนังสือพิมพ์ผู้จัดการ'}},
        {'B-LOC': {'00f78058fccdf1d1241c2c99d1a1ba37b6ad917aaedb2281c4411ba3547656d2': 'آبشار',
                   '0d3534341b5a4c6bc1187cdb23c175726222505ac7f8945f616952bc07f3c4d0': 'კუნძული',
                   '1b405f9b245ce015a282784b9b5c5beaa119d9a95f38e6273742373eb1eb3bda': "'Aklé",
                   '3e583b6bc4374a9087a29ea89c985f62fd0ec5d438fe37b56141d8c21ac867cd': 'Ķūķu',
                   '54ffe6c53880df151f1a004221e0f49116f1a0f72b094463e961bde96b406897': '"La',
                   '6ff25b9536d5242e1d4f685cee13e818fb314375a772d6817acaefe4661021c0': 'Sandys',
                   '87ea5add86dad210e6f15f20a584db1b153f578e3e93177f9d12adea7e2ad436': "'Adade",
                   '8d8387154d6b974ee4a1c82fbdc1f7e0c94da915b95d0c82dd45aa5692c11482': "'Adam",
                   'a2ac3ad896984a73e59d51963a2d76df1b88edb6e5c9a6eb9b03c8a4b63d26dd': "'Aleb",
                   'a2f85938287ad57126728554e689b0a5663be9c64aab8f70e5b4940efd81a9ef': 'تاهات',
                   'a69415cab5d699be631355dced96febfb9fb94d11b4eb679be1d4befa21c8d53': "'Achar",
                   'ab67ee23d631e05a2e7dde29e5239f8e066e1d51ce686993e559a99214b41030': 'Azeïlât',
                   'd6c5568f6d83faaedf1face638aadee00d93638b1d8fdbd62f115dfca6b3d5dd': '"Congarees"',
                   'e72d0410e669b9d23757f097dfdbf3934c9cc4b014eee204d9b5b138f0eda0b1': 'дерево',
                   'ecac331534c260df04d77b8ca77fa555783efda2b03f7f07086232f7af23a5f9': '"El',
                   'fb97ddff2c89dfbbc8bc06e67ccb3a8977f2009bd7bffe2af7597c47b374fd3c': 'Опорски'},
         'E-LOC': {'007b7c27f09b9706ef9ad7c6f02bd5d55cf28dcfa1fa2fd62d9667b29a904c5c': 'Aoukâr',
                   '156193387bc71c75f505ac653596630dcacf40eea72a4afb14065b4f86a4b1b6': 'Mountain',
                   '235124e3b7d03a294ccaa50b903e6554150a8639af9c0e1e3d2258d505062e7d': 'Cité"',
                   '23c87f1f5868d8768e766f4cc09c09b410f889e191e8f318c16e8bb0235c1392': "'Âouâna",
                   '367d7a4c5ba0fc679f0ee40592659c01feaab004de29715840277b5e72df62a1': 'Иджеване',
                   '3ae5b478e4b184fb899da66cbc341018fc0d349e9bb8757838c2530e61687aad': 'Tismana',
                   '468413cdf721bdcd2fb086dfc9d4b2c87a74ac39d04b7e93ddf50a99fa3c0e66': 'ორპირი',
                   '5899e3f73cb569aec54d7f7a58866042c9371166151466cec5c02196395dcadf': 'Dihé',
                   '6325e05fc0a7b2d45d9558c8fbbcd85338adbbfb918b8474e7e14a4d3acddbd2': "'Aïd",
                   '64e6d88c78e22d12a409ada5cdc6f9e8b0635d87aa4d429f774334aae759487e': 'Aḥmoûd',
                   '90d35e3b2406eeabe2628b746b7f35c5896a3f4e4d6116d3d62481e8941126a5': 'Teïdoûm',
                   '9a1fbfb8d07c0da58381614eaf7a1c2e3f91e3b7f4c4e68cf6870dd0faacd8d1': 'Tonatico',
                   'b3e015585ad338ee1da6c90439a934112d4403e9cd52238b5c9de04ebee46219': 'أتاكور',
                   'be8094e06a3ebbbc9979ada565f48d65349cc3008022d013ab84f91bed2a8e79': 'cliffs',
                   'c94f592e52645de0ad608b4460f9c019ae3acaec6c543310f34287c1aeec7743': 'Liyene',
                   'd4c9579ddcfb31b0fbf8959e47fe95ba857fc2d8f5eff9ea937153496125ca5a': 'شوی',
                   'db29e82a9287fc2834d8596af212c2da01149ebab1f0a48437b1f0c990af7647': 'Spring',
                   'e1f4ca76c0bf2cd8a5e2ea4e65750119af68b8be3a8aec39565fd536fbacd050': 'рид',
                   'fa7955814e32aed3a240ee46fcd053dd48f320d4e6e18d2b2774e491c5f75834': 'Site'},
         'I-LOC': {'0a9d21fe9e3a88e7a3fd82858d489e834bce82ea69c6d19952c76f76f6327de1': 'Cetățui"',
                   '23a6a75bfbaa0f9c405c5b65042f37693968624aaadb8488178152b95fdcb341': 'Bel',
                   '45fbe99b103ce035103d371a1b2d675b35b22e380d74ace82592966a3ea0f226': 'Grande',
                   '577579c5a06c525adbbd00ec167d53bf5b60416391c42f85f16ee3582bd0e2c3': 'Yus',
                   '582967534d0f909d196b97f9e6921342777aea87b46fa52df165389db1fb8ccf': 'in',
                   '6583462df78a5c131161d59457e1f5aa34e07b6a52228ec636367dd69942cb98': 'спарапетов',
                   '7bbf953b6bb0441a4afd6104f7270a7e8de38527ab72a405ddc7cf536cbbc2e3': 'archaeological',
                   'b69feea9efa8615cd491957b7ef828efb410c2dd676ca063759a689af4fef9b1': 'Fort',
                   'b8012cb642c887a0a4f6f8e52fc6d97946274076ba64b113dff1db0a0ca37caa': 'в',
                   'd65c3e892354b17a4c032593974d46bc8732c6c0b1ce56edf0c95511d1749cf5': 'el',
                   'eb52856e0b7416cfb8a976c817bd2ff69997cc18736116413ab25b1c42b72674': 'Salto"',
                   'fbae041b02c41ed0fd8a4efb039bc780dd6af4a1f0c420f42561ae705dda43fe': 'site'},
         'S-LOC': {'23da455b57dee764dcaa7291531c902d4fc48c27bdb296371eeb7af36d85dc80': 'Sandøydrætten',
                   '63e0018101b2f98a3fc7dd8061da7c60f2e0f81cfc25c49f9c183625d7d87e1a': 'Sandøygrunnen',
                   'd69e77def744c4d88cfb6e9744ef8f39776a735c80eb44181220393f5ec07ee1': 'Sandøyfjorden',
                   'ee27a9c62e5585d70dbf4c8a1bb75a3c47415775db9168094b120b17332c4af4': 'Голоух'}}]

    def setUp(self) -> None:
        self.maxDiff = None
        pass

    def test_matching_methods_good1(self):
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_mathing=True,
                                                                           partial_matching=False)
            self.assertEqual(gazetteer_embedding.matching_methods, ['full_match'])

    def test_matching_methods_good2(self):
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_mathing=False,
                                                                           partial_matching=True)
            self.assertEqual(gazetteer_embedding.matching_methods, ['partial_match'])

    def test_set_feature_list_good1(self):
        label_dict = {0: 'O',
                      1: 'B-PER', 2: 'I-PER', 3: 'E-PER', 4: 'S-PER',
                      5: 'B-ORG', 6: 'I-ORG', 7: 'E-ORG', 8: 'S-ORG',
                      9: 'B-LOC', 10: 'I-LOC', 11: 'E-LOC', 12: 'S-LOC',
                      13: 'B-MISC', 14: 'I-MISC', 15: 'E-MISC', 16: 'S-MISC'}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict,
                                                                           full_mathing=True,
                                                                           partial_matching=True)
            self.assertEqual(gazetteer_embedding.feature_list, ['O',
                                                                'B-PER', 'I-PER', 'E-PER', 'S-PER', 'B-ORG', 'I-ORG',
                                                                'E-ORG', 'S-ORG', 'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC',
                                                                'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC', 'PER', 'ORG',
                                                                'LOC', 'MISC'])
            self.assertEqual(gazetteer_embedding.embedding_length, 21)

    def test_set_feature_list_good2(self):
        label_dict = {0: 'O',
                      1: 'B-PER', 2: 'I-PER', 3: 'E-PER', 4: 'S-PER',
                      5: 'B-ORG', 6: 'I-ORG', 7: 'E-ORG', 8: 'S-ORG',
                      9: 'B-LOC', 10: 'I-LOC', 11: 'E-LOC', 12: 'S-LOC',
                      13: 'B-MISC', 14: 'I-MISC', 15: 'E-MISC', 16: 'S-MISC'}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict,
                                                                           full_mathing=False,
                                                                           partial_matching=True)
            self.assertEqual(gazetteer_embedding.feature_list, ['O',
                                                                'B-PER', 'I-PER', 'E-PER', 'S-PER', 'B-ORG', 'I-ORG',
                                                                'E-ORG', 'S-ORG', 'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC',
                                                                'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC'])
            self.assertEqual(gazetteer_embedding.embedding_length, 17)

    def test_set_feature_list_good3(self):
        label_dict = {0: 'O',
                      1: 'B-PER', 2: 'I-PER', 3: 'E-PER', 4: 'S-PER',
                      5: 'B-ORG', 6: 'I-ORG', 7: 'E-ORG', 8: 'S-ORG',
                      9: 'B-LOC', 10: 'I-LOC', 11: 'E-LOC', 12: 'S-LOC',
                      13: 'B-MISC', 14: 'I-MISC', 15: 'E-MISC', 16: 'S-MISC'}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=label_dict,
                                                                           full_mathing=True,
                                                                           partial_matching=False)
            self.assertEqual(gazetteer_embedding.feature_list, ['O', 'PER', 'ORG', 'LOC', 'MISC'])
            self.assertEqual(gazetteer_embedding.embedding_length, 5)

    def test_get_gazetteers_good(self):
        label_dict = {0: 'O',
                      1: 'B-PER', 2: 'I-PER', 3: 'E-PER', 4: 'S-PER',
                      5: 'B-ORG', 6: 'I-ORG', 7: 'E-ORG', 8: 'S-ORG',
                      9: 'B-LOC', 10: 'I-LOC', 11: 'E-LOC', 12: 'S-LOC',
                      13: 'B-MISC', 14: 'I-MISC', 15: 'E-MISC', 16: 'S-MISC'}
        with patch.object(GazetteerEmbeddings, '_process_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           label_dict=label_dict)
            self.assertEqual(gazetteer_embedding.gazetteer_file_dict_list, [{'PER': []},
                                                                            {'ORG': ['eng-ORG-alias-test.txt',
                                                                                     'eng-ORG-name-test.txt']},
                                                                            {'LOC': ['eng-LOC-name-test.txt']},
                                                                            {'MISC': []}])

    def test_process_gazetteers_good(self):
        gazetteer_files = [{'PER': []},
                           {'ORG': ['eng-ORG-alias-test.txt', 'eng-ORG-name-test.txt']},
                           {'LOC': ['eng-LOC-name-test.txt']},
                           {'MISC': []}
                           ]
        label_dict = {0: 'O',
                      1: 'B-PER', 2: 'I-PER', 3: 'E-PER', 4: 'S-PER',
                      5: 'B-ORG', 6: 'I-ORG', 7: 'E-ORG', 8: 'S-ORG',
                      9: 'B-LOC', 10: 'I-LOC', 11: 'E-LOC', 12: 'S-LOC',
                      13: 'B-MISC', 14: 'I-MISC', 15: 'E-MISC', 16: 'S-MISC'}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers', return_value=gazetteer_files), \
                patch.object(GazetteerEmbeddings, '_set_feature_list'):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="./resources",
                                                                           label_dict=label_dict)
            self.assertEqual(gazetteer_embedding.gazetteers_dicts['full_match'], self.full_match_hash_dict)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['full_match']), 2)
            # ORG with all strings as Hashes
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['full_match'][0]), 1)
            # LOC with all strings as Hashes
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['full_match'][1]), 1)

            self.assertEqual(gazetteer_embedding.gazetteers_dicts['partial_match'], self.partial_match_hash_dict)
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['partial_match']), 2)
            # ORG with B,I,S,E
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['partial_match'][0]), 4)
            # LOC with B,I,S,E
            self.assertEqual(len(gazetteer_embedding.gazetteers_dicts['partial_match'][1]), 4)

    def test_add_embeddings_internal_good1(self):
        sentences_1 = Sentence('I love Sandys Fort Spring!')
        sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
        sentence_list = [sentences_1, sentences_2]
        feature_list = ['O',
                        'B-PER', 'I-PER', 'E-PER', 'S-PER', 'B-ORG', 'I-ORG',
                        'E-ORG', 'S-ORG', 'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC',
                        'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC', 'PER', 'ORG',
                        'LOC', 'MISC']
        gazetteers = {'partial_match': self.partial_match_hash_dict, 'full_match': self.full_match_hash_dict}

        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_mathing=False,
                                                                           partial_matching=True)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # I
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[0], torch.tensor(1))

            # love
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # Sandys
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(3))
            self.assertEqual(sentence_list[0][2].embedding[5], torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[9], torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[8], torch.tensor(1))

            # Fort
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][3].embedding[10], torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[6], torch.tensor(1))

            # Spring
            self.assertEqual(torch.sum(sentence_list[0][4].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][4].embedding[11], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[7], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[0], torch.tensor(1))
            ############################################################################
            # The
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[0], torch.tensor(1))

            # Land
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[5], torch.tensor(1))

            # Tenure
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[6], torch.tensor(1))

            # Reform
            self.assertEqual(torch.sum(sentence_list[1][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][3].embedding[6], torch.tensor(1))

            # Association
            self.assertEqual(torch.sum(sentence_list[1][4].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][4].embedding[7], torch.tensor(1))

            # (
            self.assertEqual(torch.sum(sentence_list[1][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][5].embedding[0], torch.tensor(1))

            # LTRA
            self.assertEqual(torch.sum(sentence_list[1][6].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][6].embedding[0], torch.tensor(1))

            # )
            self.assertEqual(torch.sum(sentence_list[1][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][7].embedding[0], torch.tensor(1))

            # .
            self.assertEqual(torch.sum(sentence_list[1][8].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][8].embedding[0], torch.tensor(1))

    def test_add_embeddings_internal_good2(self):
        sentences_1 = Sentence('I !love! Sandys Fort Spring!')
        sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
        sentence_list = [sentences_1, sentences_2]
        feature_list = ['O',
                        'B-PER', 'I-PER', 'E-PER', 'S-PER', 'B-ORG', 'I-ORG',
                        'E-ORG', 'S-ORG', 'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC',
                        'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC', 'PER', 'ORG',
                        'LOC', 'MISC']
        gazetteers = {'partial_match': self.partial_match_hash_dict, 'full_match': self.full_match_hash_dict}

        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_mathing=True,
                                                                           partial_matching=False)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # I
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[0], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # love
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[0], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[0], torch.tensor(1))

            # Sandys
            self.assertEqual(torch.sum(sentence_list[0][4].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][4].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[19], torch.tensor(1))

            # Fort
            self.assertEqual(torch.sum(sentence_list[0][5].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][5].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[19], torch.tensor(1))

            # Spring
            self.assertEqual(torch.sum(sentence_list[0][6].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[0][6].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[0][6].embedding[19], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][7].embedding[0], torch.tensor(1))
            ############################################################################
            # The
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[0], torch.tensor(1))

            # Land
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[18], torch.tensor(1))

            # Tenure
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[18], torch.tensor(1))

            # Reform
            self.assertEqual(torch.sum(sentence_list[1][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][3].embedding[18], torch.tensor(1))

            # Association
            self.assertEqual(torch.sum(sentence_list[1][4].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][4].embedding[18], torch.tensor(1))

            # (
            self.assertEqual(torch.sum(sentence_list[1][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][5].embedding[0], torch.tensor(1))

            # LTRA
            self.assertEqual(torch.sum(sentence_list[1][6].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][6].embedding[0], torch.tensor(1))

            # )
            self.assertEqual(torch.sum(sentence_list[1][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][7].embedding[0], torch.tensor(1))

            # .
            self.assertEqual(torch.sum(sentence_list[1][8].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][8].embedding[0], torch.tensor(1))

    def test_add_embeddings_internal_good3(self):
        sentences_1 = Sentence('I !love! Sandys Fort Spring!')
        sentences_2 = Sentence('The Land Tenure Reform Association (LTRA).')
        sentences_3 = Sentence('Голоух')
        sentence_list = [sentences_1, sentences_2, sentences_3]
        feature_list = ['O',
                        'B-PER', 'I-PER', 'E-PER', 'S-PER', 'B-ORG', 'I-ORG',
                        'E-ORG', 'S-ORG', 'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC',
                        'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC', 'PER', 'ORG',
                        'LOC', 'MISC']
        gazetteers = {'partial_match': self.partial_match_hash_dict, 'full_match': self.full_match_hash_dict}
        with patch.object(GazetteerEmbeddings, '_get_gazetteers'), \
                patch.object(GazetteerEmbeddings, '_set_feature_list', return_value=feature_list), \
                patch.object(GazetteerEmbeddings, '_process_gazetteers', return_value=gazetteers):
            gazetteer_embedding: GazetteerEmbeddings = GazetteerEmbeddings(path_to_gazetteers="/path/to/gazetteers",
                                                                           label_dict=MagicMock(),
                                                                           full_mathing=True,
                                                                           partial_matching=True)
            gazetteer_embedding.embed(sentence_list)

            for sentence in sentence_list:
                for token in sentence:
                    assert len(token.get_embedding()) == len(feature_list)

            # I
            self.assertEqual(torch.sum(sentence_list[0][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][0].embedding[0], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][1].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][1].embedding[0], torch.tensor(1))

            # love
            self.assertEqual(torch.sum(sentence_list[0][2].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][2].embedding[0], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][3].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][3].embedding[0], torch.tensor(1))

            # Sandys
            self.assertEqual(torch.sum(sentence_list[0][4].embedding), torch.tensor(5))
            self.assertEqual(sentence_list[0][4].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[19], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[9], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[8], torch.tensor(1))
            self.assertEqual(sentence_list[0][4].embedding[5], torch.tensor(1))

            # Fort
            self.assertEqual(torch.sum(sentence_list[0][5].embedding), torch.tensor(4))
            self.assertEqual(sentence_list[0][5].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[19], torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[10], torch.tensor(1))
            self.assertEqual(sentence_list[0][5].embedding[6], torch.tensor(1))

            # Spring
            self.assertEqual(torch.sum(sentence_list[0][6].embedding), torch.tensor(4))
            self.assertEqual(sentence_list[0][6].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[0][6].embedding[19], torch.tensor(1))
            self.assertEqual(sentence_list[0][6].embedding[11], torch.tensor(1))
            self.assertEqual(sentence_list[0][6].embedding[7], torch.tensor(1))

            # !
            self.assertEqual(torch.sum(sentence_list[0][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[0][7].embedding[0], torch.tensor(1))
            ############################################################################
            # The
            self.assertEqual(torch.sum(sentence_list[1][0].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][0].embedding[0], torch.tensor(1))

            # Land
            self.assertEqual(torch.sum(sentence_list[1][1].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][1].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[1][1].embedding[5], torch.tensor(1))

            # Tenure
            self.assertEqual(torch.sum(sentence_list[1][2].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][2].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[1][2].embedding[6], torch.tensor(1))

            # Reform
            self.assertEqual(torch.sum(sentence_list[1][3].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][3].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[1][3].embedding[6], torch.tensor(1))

            # Association
            self.assertEqual(torch.sum(sentence_list[1][4].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[1][4].embedding[18], torch.tensor(1))
            self.assertEqual(sentence_list[1][4].embedding[7], torch.tensor(1))

            # (
            self.assertEqual(torch.sum(sentence_list[1][5].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][5].embedding[0], torch.tensor(1))

            # LTRA
            self.assertEqual(torch.sum(sentence_list[1][6].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][6].embedding[0], torch.tensor(1))

            # )
            self.assertEqual(torch.sum(sentence_list[1][7].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][7].embedding[0], torch.tensor(1))

            # .
            self.assertEqual(torch.sum(sentence_list[1][8].embedding), torch.tensor(1))
            self.assertEqual(sentence_list[1][8].embedding[0], torch.tensor(1))
            ############################################################################
            # Голоух
            self.assertEqual(torch.sum(sentence_list[2][0].embedding), torch.tensor(2))
            self.assertEqual(sentence_list[2][0].embedding[12], torch.tensor(1))
            self.assertEqual(sentence_list[2][0].embedding[19], torch.tensor(1))
