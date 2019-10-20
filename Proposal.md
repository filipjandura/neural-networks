# Rozpoznávanie objektov v datasete malých obrázkov
  Dataset CIFAR *(Canadian Institute For Advanced Research)* \
  <https://www.cs.toronto.edu/~kriz/cifar.html>
### Návrh zadania

Navrhnite postup spracovavnia obrazov a architektúru siete, pomocou ktorej budete vedieť rozpoznávať objekty na malých obrázkoch, ktoré sa nachádzajú v datasete CIFAR, či už v alternatíve 10 ale aj 100 tried. 

### Motivácia
Rozpoznávanie objektov z obrazu sa stretáva s každodenným využitím. Už historicky najbežnejšie požiadavky na systémy boli rozpoznávanie tvárí človeka, automatické rozpoznanie čísiel či už na bankomatových/vernostných kartách alebo ŠPZ vozidla. Aj pri náročnejších problémoch rozpoznávania objektov, ktoré boli pomocou klasických prístupov neriešiteľné, sa vďaka rýchlemu vývoju v oblasti hlbokého učenia a príchodom neurónových sietí začínajú dosahovať výsledky doahujúce, v niektorých prípadoch dokonca presahujúce presnosť človeka. Najbežnejšie sa stretávame s rozpoznávaním objektov a osôb na sociálnych sieťach, kde algoritmy priamo rozpoznávajú ľudí na fotografiách. Veľmi veľký význam a prínos má rozpoznávanie objektov pre vyhľadávacie stroje, ktoré po analýze vstupného obrazu, odporúčajú podobné obrazy s nájdenými objektami. 

Napriek úspešnosti a neporovnateľne lepším výsledkom voči klasickým metódam, neurónové siete trpia viacerými problémami, ktoré obmedzujú ich rozsiahle nasadenie v reálnej praxi a bežnej prevádzke. Jedným z najväčším problémom pri rozpoznávaním objektov v obrazoch je otázka preučenia sa na trénovacích dátach a teda nedostatočného fungovania počas testovania a v prevádzke. Ak sieť priveľa krát uvidí 100 stoličiek, síce sa naučí ich generalizovanú reprezentáciu, je ale možné, že nebude vedieť rozpoznať stoličku stoprvú.

Ďalším nedostatkom je problém pri trénovaní. Pre dostatočné naučenie sa rozpoznávania veľkého množstva objektov, je štandardne nutné aby bola sieť dostatočne hlboká a obsahovala dostatočné množstvo naučiteľných parametrov. Tréning takto rozsiahlych a hlbokých siete môže trvať pridlho a nemusí sa skončiť úspešne. Na druhej strane tiež dochádza k problémovému nasadzovaniu pre praktické použitie. 

### Existujúce State-of-the-art riešenia
Dataset CIFAR sa svojou podstatou a relatívnou náročnosťou stáva akýmsi benchmarkom pre neurónové siete, ktoré sú zamerané na rozpoznávanie objektov. Či už jednoduchšia alternatíva s 10 kategóriami alebo náročnejšia, pri ktorej je potrebné klasifikovať 100 tried objektov reálneho sveta. Dataset vytvoril Alex Krizhevsky, Vinod Nair a Geoffrey Hinton. Od vtedy bol tento dataset riešený v rôznych výskumoch. Až 17 zo state-of-the-art [registrovaných] výsledkov dosahuje úspešnosť klasifikácie lepšiu ako 80%, z toho len 2 dosahujú úspešnosť vyššiu ako 90%.

Všetky state-of-the-art riešenia používajú hlboké neurónové siete. Základná architektúra, ktorú všetky práce rozširujú sú konvolučné neurónové siete. Najčastejšie používané prístupy sú [DenseNet], [ResNet] a ich obmeny. DenseNet stavia na poznatku, že konvolučné siete, môžu byť omnoho hlbšie, presnejšie a ľahšie na trénovanie pokiaľ obsahujú kratšie spojenia medzi vrstvami bližšie k vstupu a k výstupu. 

DenseNet spája každé dve po sebe idúce vrstvy feed-forward spôsobom a teda naproti klasickým sieťam, ktoré majú prepojenia len medzi za sebou idúcimi vrstvami, DenseNet dáva pre každú vrstvu do vstupu mapy príznakov zo všetkých predchádzajúcich vrstiev. ResNet obsahuje reziduálne skip bloky, ktoré pomáhajú sieti zachovať presnosť, aby nenarastala chyba spôsobená aproximovaním pri veľmi hlbokých sieťach. 

[WideResNet] sa pokúša riešiť problém s efektívnosťou reziduálnych sietí, ktoré kvôli zvýšeniu presnosti aj keď o 1% musia narásť parametrami takmer na dvojnásobok. Riešenie ponúkajú v možnosti nárastu siete do šírky a obmedzení hĺbky, a ich aj keď len 16 vrstvové siete prekonávajú úspešnosť často tenkých a hlbokých reziduálnych sietí.

Experimentálny prístup zvolili tvorcovia [SimpleNet]u, ktorí zvolili opačnú taktiku ako predchádzajúce prístupy a nesnažili sa vytvoriť čo najrobustnejšiu a najväčšiu sieť. SimpleNet či už vo verzií 1 ale aj 2 (SimpNet) je jednoduchá sieť, ktorá je špeciálne dizajnovaná, aby bola postačujúca pre úlohy, na ktoré je určená. 

Druhé najúspešnejšie riešenie je GPipe s 91.3% na [CIFAR-100], ktoré je vlastne rozsiahla paralelizačná knižnica, ktorá umožňuje trénovanie gigantických hlbokých sietí. Toto riešenie je však pri bežných podmienkach takmer nemožné využiť a replikovať.

Najnovší príspevok je [EfficientNet], kde sa autori pokúšajú využiť možnosti postupného zväčšovania a spresňovania nájsť najlepšiu - najefektívnejšiu architektúru pre danú úlohu.

### Podobné datasety
<http://deeplearning.net/datasets/> \
Microsoft COCO, ImageNet, MNIST-Fashion, Caltech 101/256, Pascal VOC, SVHN, ... a veľa ďalších.
Keďže problematika rozpoznávania objektov je v oblasti počítačového videnia riešená už niekoľko rokov, aj pred príchodom neurónových sietí, patrične k tomu existuje aj veľké množstvo datasetov, ktoré sú generické ale aj špecializované na túto úlohu.

Dataset CIFAR je zameraný na rozpoznávanie objektov. Obe varianty majú rovnaké vlastnosti. Obrázky veľkosti 32x32x3 RGB s 8bitovou hĺbkou na kanál. Rozdelenie tried je pravidelné, 600 obrázkov na triedu - 500 trénovacích 100 testovacích. Celkovo teda 6000 obrázkov pre CIFAR-10 a 60000 obrázkov pre CIFAR-100.

ImageNet, COCO a Pascal VOC sú datasety obsahujúce obrázky reálneho sveta, rôznych rozmerov, používajú sa pri náročnejších úlohách rozpoznávania objektov z reálnej scény. Obsahujú niekoľko 1000 tried a každá trieda má iný počet obrázkov, spolu niekoľko miliónov obrázkov v celom datasete.

MNIST-Fashion je testovací dataset s náročnejším vstupom, obrázky rozmerov 28x28x1 odtiene šedej. Obsahuje 10 tried typov oblečenia, pre každú triedu 500 trénovacích a 100 testovacích vzoriek. 

### Návrh metódy a možný postup
Existujúca sieť [SimpleNet] ponúka optimálne a efektívne riešenie pre problém rozpoznávania objektov z malých obrázkov. Pre prvé experimenty by sme použili základnú architektúru, ktorá bola prezentovaná a registrovaná ako state-of-the-art pre [CIFAR-10] (v1-95.51%/v2-96.26%) aj [CIFAR-100] (v1-78.37%/v2-80.29%).

 
[registrovaných]: https://paperswithcode.com/sota/image-classification-on-cifar-100
[DenseNet]: https://paperswithcode.com/paper/densely-connected-convolutional-networks
[ResNet]: https://paperswithcode.com/paper/wide-residual-networks
[WideResNet]: https://paperswithcode.com/paper/wide-residual-networks
[SimpleNet]: https://paperswithcode.com/paper/lets-keep-it-simple-using-simple
[EfficientNet]: https://paperswithcode.com/paper/efficientnet-rethinking-model-scaling-for
[CIFAR-10]: https://paperswithcode.com/sota/image-classification-on-cifar-10
[CIFAR-100]: https://paperswithcode.com/sota/image-classification-on-cifar-100