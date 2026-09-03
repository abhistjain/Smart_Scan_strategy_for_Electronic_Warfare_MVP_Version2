"use client";

// Shared Plotly component built on the lightweight dist-min bundle via the
// react-plotly.js factory. Imported through next/dynamic (ssr:false) by callers
// so Plotly never runs on the server.
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);
export default Plot;
