// The dist-min bundle ships no types; the factory only needs the default export.
declare module "plotly.js-dist-min" {
  const Plotly: any;
  export default Plotly;
}
