# Lizenz der Daten in `toronto_campus.json`

Die Datei `toronto_campus.json` enthält einen **Auszug aus OpenStreetMap**.

© OpenStreetMap-Mitwirkende – Daten unter der **Open Database License (ODbL) 1.0**:
<https://opendatacommons.org/licenses/odbl/1-0/> · Hinweise zur Namensnennung: <https://www.openstreetmap.org/copyright>

**Was der Auszug ist:** das Straßen- und Wegenetz im Umkreis von 1 000 m um den Mittelpunkt des St. George Campus der University of Toronto (43.662643, -79.395689),
alle Wegarten, abgefragt am 21.09.2026 über die Overpass-Schnittstelle mit `osmnx` (`tools/fetch_osm.py`).

**Was daran verändert wurde:** die Straßen sind zu Knoten und gerichteten Kanten vereinfacht (osmnx-Vereinfachung), Parallelkanten sind auf die kürzeste reduziert, Selbstschleifen entfernt,
die Koordinaten sind lokale Meter relativ zum Mittelpunkt, die Kantenlängen in Metern. Start (Reiterstandbild König Eduards VII.) und Ziel (Bahen Centre) sind die nächsten Knoten zu den im Skript angegebenen Koordinaten.

**Was das für Sie bedeutet:** die Datei darf unter den Bedingungen der ODbL weitergegeben, verändert und genutzt werden – mit Namensnennung (siehe oben) und
**Share-Alike**: wer sie oder eine daraus abgeleitete Datenbank öffentlich weitergibt, muss das unter der ODbL tun. Diese Lizenz gilt nur für die Daten in dieser Datei,
nicht für den übrigen Inhalt des Repositories.
