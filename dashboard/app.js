"use strict";

const colors = {high: "#d94c4c", medium: "#d99500", low: "#22a06b", unknown: "#8793a0"};
const risk = value => Object.hasOwn(colors, value) ? value : "unknown";
const finite = value => typeof value === "number" && Number.isFinite(value);
const validPosition = vehicle => vehicle.location_valid && finite(vehicle.lat) && finite(vehicle.lon);
const probabilityText = value => finite(value) ? `${Math.round(value * 100)}%` : "неизвестно";
const delayText = value => finite(value) ? `${value} мин` : "нет прогноза";
const speedText = value => finite(value) ? `${Number(value).toFixed(1)} км/ч` : "неизвестна";
let map, routeSource, vehicleSource, incidentSource, refreshing = false, didZoom = false;

function textElement(tag, text, className) {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    return element;
}

function vehicleCard(vehicle) {
    const card = textElement("div", "", "item");
    card.append(textElement("b", vehicle.tr_id || vehicle.unit_id || "Неизвестное ТС"));
    card.append(textElement("div", vehicle.scheduled ? "По расписанию" : "Контекстное ТС", "muted"));
    card.append(textElement("div", `Скорость: ${speedText(vehicle.speed)}`));
    if (vehicle.position_stale) card.append(textElement("div", "Показана последняя известная позиция", "muted"));
    if (vehicle.forecast) {
        const forecast = vehicle.forecast;
        card.append(textElement(
            "div", `Риск ${probabilityText(forecast.delay_probability)}, прогноз ${delayText(forecast.delay_minutes)}`,
            risk(forecast.risk),
        ));
        if (forecast.degraded) card.append(textElement("div", "Режим деградации: вероятность неизвестна", "muted"));
    }
    if (validPosition(vehicle)) {
        card.addEventListener("click", () => map?.easeTo({
            center: [vehicle.lon, vehicle.lat], zoom: 15, duration: 250,
        }));
    }
    return card;
}

async function fetchJson(path) {
    const response = await fetch(path, {signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

function emptyFeatureCollection() {
    return {type: "FeatureCollection", features: []};
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, character => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
    }[character]));
}

function mapUnavailable(message) {
    const target = document.querySelector("#map");
    target.replaceChildren(textElement("div", message, "map-message"));
}

function initializeMap() {
    if (!window.maplibregl || !window.maplibregl.Map) {
        mapUnavailable("MapLibre не загрузился. Проверьте подключение к библиотеке карт.");
        return;
    }
    map = new window.maplibregl.Map({
        container: "map",
        center: [37.62, 55.75],
        zoom: 10,
        style: {
            version: 8,
            sources: {
                "osm-raster": {
                    type: "raster",
                    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                    tileSize: 256,
                    attribution: "© OpenStreetMap contributors",
                },
            },
            layers: [{id: "osm-raster-layer", type: "raster", source: "osm-raster"}],
        },
    });
    map.addControl(new window.maplibregl.NavigationControl(), "top-right");
    map.on("load", () => {
        routeSource = "official-routes";
        vehicleSource = "vehicles";
        incidentSource = "incidents";
        map.addSource(routeSource, {type: "geojson", data: emptyFeatureCollection()});
        map.addSource(vehicleSource, {type: "geojson", data: emptyFeatureCollection()});
        map.addSource(incidentSource, {type: "geojson", data: emptyFeatureCollection()});
        map.addLayer({
            id: "official-routes-halo", type: "line", source: routeSource,
            layout: {"line-cap": "round", "line-join": "round"},
            paint: {"line-color": "#ffffff", "line-width": 7, "line-opacity": .38},
        });
        map.addLayer({
            id: "official-routes-layer", type: "line", source: routeSource,
            layout: {"line-cap": "round", "line-join": "round"},
            paint: {"line-color": ["get", "color"], "line-width": 3.5, "line-opacity": .9},
        });
        map.addLayer({
            id: "vehicles-layer", type: "circle", source: vehicleSource,
            paint: {
                "circle-radius": ["case", ["get", "scheduled"], 7, 5],
                "circle-color": ["get", "color"], "circle-stroke-color": "#fff",
                "circle-stroke-width": 1.2, "circle-opacity": .9,
            },
        });
        map.addLayer({
            id: "incidents-layer", type: "circle", source: incidentSource,
            paint: {
                "circle-radius": 10, "circle-color": ["get", "color"],
                "circle-stroke-color": "#fff", "circle-stroke-width": 2,
                "circle-opacity": .95,
            },
        });
        map.on("click", "vehicles-layer", event => {
            const properties = event.features?.[0]?.properties || {};
            const content = document.createElement("div");
            content.append(textElement("strong", properties.label || "ТС"));
            content.append(textElement("div", properties.description || ""));
            content.append(textElement("div", properties.speed || ""));
            new window.maplibregl.Popup().setLngLat(event.lngLat).setDOMContent(content).addTo(map);
        });
        map.on("mouseenter", "vehicles-layer", () => { map.getCanvas().style.cursor = "pointer"; });
        map.on("mouseleave", "vehicles-layer", () => { map.getCanvas().style.cursor = ""; });
        map.on("click", "incidents-layer", event => {
            const properties = event.features?.[0]?.properties || {};
            const content = document.createElement("div");
            content.append(textElement("strong", properties.label || "Инцидент"));
            content.append(textElement("div", properties.description || ""));
            new window.maplibregl.Popup().setLngLat(event.lngLat).setDOMContent(content).addTo(map);
        });
        map.on("mouseenter", "incidents-layer", () => { map.getCanvas().style.cursor = "pointer"; });
        map.on("mouseleave", "incidents-layer", () => { map.getCanvas().style.cursor = ""; });
        refresh();
    });
}

function lineFeature(points, color) {
    return {
        type: "Feature",
        properties: {color},
        geometry: {type: "LineString", coordinates: points.map(([lat, lon]) => [lon, lat])},
    };
}

function vehicleFeature(vehicle, color) {
    return {
        type: "Feature",
        properties: {
            label: vehicle.tr_id || vehicle.unit_id || "ТС",
            scheduled: Boolean(vehicle.scheduled),
            color,
            speed: `Скорость: ${speedText(vehicle.speed)}`,
            description: `${vehicle.scheduled ? "По расписанию" : "Контекстное ТС"}${
                vehicle.position_stale ? " · последняя известная позиция" : ""
            }`,
        },
        geometry: {type: "Point", coordinates: [vehicle.lon, vehicle.lat]},
    };
}

function incidentFeature(incident) {
    const position = incident.position || {};
    return {
        type: "Feature",
        properties: {
            label: `${incident.tr_id || "ТС"} — инцидент`,
            color: colors[risk(incident.risk)],
            speed: `Скорость: ${speedText(incident.speed)}`,
            description: `Риск ${probabilityText(incident.delay_probability)} · ${
                incident.reason || "предупреждающий сигнал"
            }`,
        },
        geometry: {type: "Point", coordinates: [position.lon, position.lat]},
    };
}

function renderMap(routes, vehicles, incidents) {
    if (!map || !routeSource || !vehicleSource || !incidentSource) return;
    const currentById = new Map(vehicles.filter(item => item.tr_id).map(item => [item.tr_id, item]));
    const coordinates = [];
    const routeFeatures = [];
    const vehicleFeatures = [];
    const incidentFeatures = [];
    for (const route of routes) {
        const points = route.stops.filter(stop => finite(stop.lat) && finite(stop.lon))
            .map(stop => [stop.lat, stop.lon]);
        if (!points.length) continue;
        coordinates.push(...points.map(([lat, lon]) => [lon, lat]));
        if (points.length > 1) {
            const routeRisk = risk(currentById.get(route.tr_id)?.forecast?.risk);
            routeFeatures.push(lineFeature(points, colors[routeRisk]));
        }
    }
    for (const vehicle of vehicles) {
        if (!validPosition(vehicle)) continue;
        const color = vehicle.scheduled ? colors[risk(vehicle.forecast?.risk)] : "#3d8ee8";
        vehicleFeatures.push(vehicleFeature(vehicle, color));
        coordinates.push([vehicle.lon, vehicle.lat]);
    }
    for (const incident of incidents.slice(-120)) {
        const position = incident.position || {};
        if (!finite(position.lat) || !finite(position.lon)) continue;
        incidentFeatures.push(incidentFeature(incident));
    }
    map.getSource(routeSource).setData({type: "FeatureCollection", features: routeFeatures});
    map.getSource(vehicleSource).setData({type: "FeatureCollection", features: vehicleFeatures});
    map.getSource(incidentSource).setData({
        type: "FeatureCollection", features: incidentFeatures,
    });
    if (coordinates.length && !didZoom) {
        const bounds = coordinates.reduce(
            (result, coordinate) => result.extend(coordinate),
            new window.maplibregl.LngLatBounds(coordinates[0], coordinates[0]),
        );
        map.fitBounds(bounds, {padding: 30, maxZoom: 13, duration: 250});
        didZoom = true;
    }
}

async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
        const [schedule, state] = await Promise.all([
            fetchJson("/api/schedule"), fetchJson("/api/state"),
        ]);
        renderMap(schedule.routes, state.vehicles, state.incidents);
        document.querySelector("#scheduled").textContent = state.vehicles.filter(validPosition).length;
        document.querySelector("#alerts").textContent = state.incidents.length;
        document.querySelector("#packets").textContent = state.metrics.packets;
        const mae = finite(state.metrics.measured_absolute_error_seconds) && state.metrics.measured_error_count
            ? `${(state.metrics.measured_absolute_error_seconds / state.metrics.measured_error_count).toFixed(1)} с`
            : "ожидает фактических прибытий";
        document.querySelector("#status").textContent =
            `API доступен · ${new Date().toLocaleTimeString()} · online MAE: ${mae}`;

        const incidents = state.incidents.slice(-60).reverse().map(incident => {
            const card = textElement("div", "", "item");
            card.append(textElement("b",
                `${incident.tr_id || "ТС"} — риск ${probabilityText(incident.delay_probability)}`,
                risk(incident.risk)));
            card.append(textElement("div",
                `Прогноз: ${delayText(incident.delay_minutes)} · ${incident.reason || "причина не определена"}`));
            card.append(textElement("div",
                `Остановка ${incident.target_stop_id || "—"} · сегмент ${incident.problem_segment?.segment_index ?? "—"}`,
                "muted"));
            card.append(textElement("div", `Скорость: ${speedText(incident.speed)}`, "muted"));
            card.append(textElement("div", incident.recommendation || "", "muted"));
            card.append(textElement("div", incident.updated_at || incident.measured_at || "", "muted"));
            if (finite(incident.position?.lat) && finite(incident.position?.lon)) {
                card.addEventListener("click", () =>
                    map?.easeTo({center: [incident.position.lon, incident.position.lat], zoom: 15, duration: 250}));
            } else {
                card.append(textElement("div", "Координаты инцидента недоступны", "muted"));
            }
            return card;
        });
        document.querySelector("#incidents").replaceChildren(
            ...(incidents.length ? incidents : [textElement("span", "Нет инцидентов", "muted")]),
        );
        const vehicleCards = state.vehicles.slice(-40).reverse().map(vehicleCard);
        document.querySelector("#vehicles").replaceChildren(
            ...(vehicleCards.length ? vehicleCards : [textElement("span", "Ожидание телеметрии…", "muted")]),
        );
    } catch (error) {
        document.querySelector("#status").textContent = `Ошибка связи с API: ${error.message}`;
    } finally {
        refreshing = false;
    }
}

function initializeTheme() {
    const stored = localStorage.getItem("transit-theme");
    const theme = stored === "dark" || stored === "light" ? stored : "light";
    document.documentElement.dataset.theme = theme;
    updateThemeButton(theme);
    document.querySelector("#theme-toggle").addEventListener("click", () => {
        const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        localStorage.setItem("transit-theme", next);
        updateThemeButton(next);
    });
}

function updateThemeButton(theme) {
    document.querySelector("#theme-toggle").textContent =
        theme === "dark" ? "Светлая тема" : "Тёмная тема";
}

document.querySelector("#refresh").addEventListener("click", refresh);
document.querySelector("#whatif-run").addEventListener("click", async () => {
    const output = document.querySelector("#whatif-result");
    output.textContent = "Расчёт…";
    try {
        const response = await fetch("/api/what-if", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                tr_id: document.querySelector("#whatif-tr").value.trim() || null,
                route_id: null,
                extra_vehicles: Number(document.querySelector("#whatif-extra").value),
            }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
        output.textContent = `ТС: ${result.observed_scheduled_vehicles} → ${result.projected_vehicles}; ` +
            `интервал: ${result.estimated_headway_before_minutes} → ${result.estimated_headway_after_minutes} мин. ` +
            result.assumptions.join(" ");
    } catch (error) {
        output.textContent = `Ошибка сценария: ${error.message}`;
    }
});

function waitForMapLibre(attempts = 0) {
    if (window.maplibregl?.Map) {
        initializeMap();
        return;
    }
    if (attempts >= 50) {
        mapUnavailable("MapLibre не загрузился. Проверьте доступ к CDN.");
        return;
    }
    window.setTimeout(() => waitForMapLibre(attempts + 1), 100);
}

initializeTheme();
waitForMapLibre();
refresh();
setInterval(refresh, 5000);
