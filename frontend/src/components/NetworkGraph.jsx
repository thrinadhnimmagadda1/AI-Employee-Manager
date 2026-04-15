import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useNavigate } from "react-router-dom";

export default function NetworkGraph({ graphData }) {
  const ref = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!graphData || !graphData.nodes?.length) return;
    const { nodes, links } = graphData;

    const container = ref.current;
    const W = container.clientWidth || 700;
    const H = 420;

    // Clear previous render
    d3.select(container).selectAll("*").remove();

    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", W)
      .attr("height", H)
      .attr("viewBox", `0 0 ${W} ${H}`)
      .style("background", "transparent");

    // Tooltip
    const tooltip = d3
      .select(container)
      .append("div")
      .style("position", "absolute")
      .style("background", "#111827")
      .style("border", "1px solid #374151")
      .style("border-radius", "8px")
      .style("padding", "8px 12px")
      .style("font-size", "12px")
      .style("color", "#fff")
      .style("pointer-events", "none")
      .style("opacity", 0);

    // Clone for simulation
    const simNodes = nodes.map((n) => ({ ...n }));
    const simLinks = links.map((l) => ({
      ...l,
      source: simNodes.findIndex((n) => n.id === l.source),
      target: simNodes.findIndex((n) => n.id === l.target),
    })).filter((l) => l.source >= 0 && l.target >= 0);

    const simulation = d3
      .forceSimulation(simNodes)
      .force("link", d3.forceLink(simLinks).distance(80).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide().radius((d) => d.size + 4));

    // Edges
    const link = svg
      .append("g")
      .selectAll("line")
      .data(simLinks)
      .enter()
      .append("line")
      .attr("stroke", (d) => d.color || "#3b82f6")
      .attr("stroke-width", (d) => d.thickness || 1.5)
      .attr("stroke-opacity", 0.6);

    // Edge hover tooltip
    link.on("mouseover", (event, d) => {
      tooltip
        .style("opacity", 1)
        .html(
          `Relationship health: ${((d.relationship_health || 0.5) * 100).toFixed(0)}%<br/>
           Messages: ${d.weight}<br/>
           Avg sentiment: ${(d.avg_sentiment || 0).toFixed(2)}`
        );
    })
    .on("mousemove", (event) => {
      tooltip
        .style("left", event.offsetX + 12 + "px")
        .style("top", event.offsetY - 28 + "px");
    })
    .on("mouseout", () => tooltip.style("opacity", 0));

    // Nodes
    const node = svg
      .append("g")
      .selectAll("circle")
      .data(simNodes)
      .enter()
      .append("circle")
      .attr("r", (d) => d.size || 10)
      .attr("fill", (d) => d.color || "#4f6ef7")
      .attr("stroke", "#111827")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      )
      .on("click", (_, d) => navigate(`/employee/${d.id}`))
      .on("mouseover", (event, d) => {
        tooltip
          .style("opacity", 1)
          .html(
            `${d.name}<br/>
             Score: ${d.overall_score ?? "—"}/100<br/>
             Burnout: ${((d.burnout_risk || 0) * 100).toFixed(0)}%<br/>
             Msgs: ${d.message_count}`
          );
      })
      .on("mousemove", (event) => {
        tooltip
          .style("left", event.offsetX + 12 + "px")
          .style("top", event.offsetY - 28 + "px");
      })
      .on("mouseout", () => tooltip.style("opacity", 0));

    // Labels
    const label = svg
      .append("g")
      .selectAll("text")
      .data(simNodes)
      .enter()
      .append("text")
      .text((d) => (d.name || "").slice(0, 8))
      .attr("font-size", 9)
      .attr("fill", "#9ca3af")
      .attr("text-anchor", "middle")
      .attr("dy", (d) => -(d.size || 10) - 4);

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
      label.attr("x", (d) => d.x).attr("y", (d) => d.y);
    });

    return () => simulation.stop();
  }, [graphData, navigate]);

  if (!graphData?.nodes?.length) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-600 text-sm">
        No graph data available for this week.
      </div>
    );
  }

  return (
    <div
      ref={ref}
      className="relative w-full rounded-xl overflow-hidden bg-gray-950/50"
      style={{ minHeight: 420 }}
    />
  );
}
