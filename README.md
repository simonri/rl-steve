# Self-Taught Steve: Steve lär sig Minecraft själv med reinforcement learning

Parkour, bridging och snart Bedwars, i riktiga Minecraft 26.3 från speltillstånd och utan pixlar. Steve tränar i en egen simulator skriven i Rust.

- `mod/`: Fabric-mod som stegar klient och inbyggd server ett tick i taget från Python.
- `python/`: anslutning, Gymnasium-miljö, träning, uppspelning, dashboard och tester.
- `sim/`: Minecrafts spelarfysik och träningsrundorna i Rust, för att träna utan spelet (se nedan).
- `runs/`: spelkataloger, loggar och tränade modeller.

## Träna i simulatorn (det vanliga nu)
`sim/` gör samma sak som spelet för spelaren: rörelse, kollision, hopp, sprint, shift vid kanten, sikte, placering och
fallskada, följt rad för rad efter den dekompilerade 26.3-koden, och sedan `AutoEnv`:s regler, poäng och de 4681 talen.
Banorna ritas av samma Python-kod som förut (`sim_env.CourseMaker`). 512 rundor körs sida vid sida på alla kärnor och
modellen väljer drag åt alla på grafikkortet, ungefär 5 000 beslut per sekund mot 420 i spelet. Modellen den sparar spelar
det riktiga spelet utan ändringar.

    cd sim && PYO3_PYTHON=$(uv python find 3.12) cargo build --release && cp target/release/libmcsim.so ../python/mcsim.so
    cd ../python && $U train_sim.py --name auto6 --resume ../runs/models/auto5/latest.zip >> ../runs/auto6.log 2>&1 &
    $U live_auto.py --name auto6 --port 25601 > ../runs/live.log 2>&1 &   # visning på riktig server, localhost:25701

Utan `--resume` fortsätter den körningen från sin egen `latest.zip`. Kontroller:
- `sim_traces.py` spelar in vad det riktiga spelet gör tick för tick, `sim_check.py` spelar upp det i simulatorn: 99 % av
  ticken blir exakt lika, resten är mest spelets egna kapplöpningar mellan klient och server.
- `test_sim_env.py` kör den oförändrade `AutoEnv` på simulatorn bredvid Rust-rundorna: allt den ser och alla poäng ska vara lika.
- `sim_bench.py` mäter simulatorn ensam (30 000–45 000 beslut per sekund med slumpdrag).

## Krav
Java 25 (`mise install java@temurin-25`) och `uv`. Bygg en gång, och efter varje ändring i modet:

    cd mod && JAVA_HOME=$(mise where java@temurin-25) ./gradlew writeClientCommand

## Starta allt (från `python/`)
Spelinstanserna öppnas på workspace 19:

    python3 launch.py bot0 --port 25601 --watch-port 25701   # visning
    python3 launch.py bot1 --port 25602 --watch-port 25702   # träning, går att titta på
    python3 launch.py bot2 --port 25603                      # träning
    python3 launch.py bot3 --port 25606                      # träning
    python3 launch.py bot4 --port 25607                      # träning

    U="uv run --no-project --python 3.12 --with torch --with stable-baselines3 --with sb3-contrib --with gymnasium python"

Träningen lär sig på grafikkortet om det finns ett (torch från PyPI har stöd för det) och spelar på processorn.
    $U train_auto.py --name auto5 --start-from-recordings 3000 --recordings ../runs/demos ../runs/demos_teacher >> ../runs/auto5.log 2>&1 &
    $U live_auto.py --name auto5 --port 25601 > ../runs/live.log 2>&1 &
    python3 dashboard.py --port 8791 > ../runs/dashboard.log 2>&1 &

Fortsätt en träning med `--resume ../runs/models/auto5/latest.zip`. Nivåerna läses från `levels.json` i samma mapp.
Stoppa en spelinstans med `./stop_client.sh bot0`.

## Så väljs banorna
`curriculum.py` har alla banor som reglage per banatyp (parkour, väg, bro, Treenan). Ett reglage höjs vid 70 % rundor
utan död och sänks vid 25 %. De allra svåraste reglagen öppnas först när resten av banatypen är på max, och de snabbaste
broarna jagar tid tills spelet självt sätter gränsen. `auto_env.py` bygger banorna och ser allt från spelarens blick.
`skill_env.py` med `train_skill.py` och `live.py` är den gamla stegvisa träningen (körningen `skill8`).

## Titta
- **Lunar eller annan 26.3-klient:** `localhost:25702` visar träningen. `localhost:25701` visar senaste
  modellen, och den spelar bara när någon är inne.
- **Dashboard:** https://simon-1.tailca3fd9.ts.net:5443/ (bara i ditt tailnet).
  Vägen sattes upp med `tailscale serve --bg --https=5443 http://127.0.0.1:8791`.
  Stäng den med `tailscale serve --https=5443 off`.

## Testserver på Treenan-kartan
En egen spelinstans där den som ansluter väljer uppgift: föremål 1 bygger bedön och mittön, föremål 2 bedön och en
sidoö diagonalt. Gapet slumpas mellan 10 och 30 block, och hela området rensas först. Ingen bridgear här än.

    python3 launch.py maptest --port 25604 --watch-port 25704 --test-menu --render-distance 12
    $U map_test.py --port 25604 > ../runs/map_test.log 2>&1 &

Anslut till `localhost:25704`. Stoppa med `./stop_client.sh maptest` och genom att avsluta `map_test.py`.

## Spela in riktig data
En egen spelinstans där den som ansluter får slumpade rundor: parkour, vägar, broar och Treenan-kartan, med alla reglage slumpade. Bara klarade rundor
sparas, bara det lyckade försöket, och väntan på en halv sekund eller mer klipps bort.

    python3 launch.py demo --port 25608 --watch-port 25708 --human
    $U demo_server.py --port 25608 > ../runs/demo.log 2>&1 &

Som hjälp vid bro finns `mod/build/libs/mcbridge-helpers-0.2.0.jar` (bygg med `./gradlew autoBridgeJar`), för mods-mappen i en
Fabric-klient. G slår på och av båda: shift vid kanten medan du själv går och siktar, och klick bara medan du håller höger musknapp,
med shift nere eller precis släppt. Bara med block i handen och bara på en server på den här datorn.

Anslut till `localhost:25708`. Rundorna hamnar i `runs/demos/`. `$U demo_data.py` gör om dem till det modellen ser och
vad den hade tryckt, i `runs/demos.npz`. Testa med `test_demo.py` och en andra klient som spelare.

## Tester
`test_auto.py`, `test_rewards.py`, `test_human_limits.py` och `test_skill_env.py` kräver inget spel. Övriga `test_*.py` körs mot en
egen spelinstans, till exempel `$U test_respawn.py 25605`. En instans tar bara en Python-anslutning åt gången, så använd
inte portarna som träningen eller `live.py` kör på. `diagnose_*.py` mäter vad en sparad modell gör, utan att träna.
