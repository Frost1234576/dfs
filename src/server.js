const fastify = require('fastify')();

const extreme = ["xFoodCatx","Ryguygamer282","Quizzled","JetsArmy", "Desstyer", "Tryerna", "JJJT", "Teacat1", "marisa555", "Tdad15", "BoomyBeats", "adrianweishaar", "WaterDragonEyes", "CoolDudeSorin", "Major_Games", "4lwn", "Tmobar", "LordMiguel", "wodozar", "CharlotteWlt", "Tqlted", "HighliteLowlite", "Matryx526", "MrHydeALoogie", "iKinglish", "Ducklat", "spamtonthetaxman", "Raw_Cod", "FriendlyGecko97", "SourThread20399", "nvct", "NuggetZealot", "HomelessReject", "FRANCIDIUM", "DetectiveDru", "tomato_GOD_ig", "Chickn_Fries", "BFsLego", "RheithEd", "Magnus6109", "EagleAri", "BlueAdventure13", "The_Thwoop", "AxEolotSof", "robottraveler", "shahardel90", "BozonZ", "Gr8Gatsby4625", "Josforce25", "bagelcracker", "Prackledd", "Cringetastic", "Harzt"]
const minor = ["DEADLY_JOHN"];
const whitelisted = ["Durmatt","Chatham_Pigeon","Frostbqte"]

fastify.get("/banlist", async (req, reply) => {
  try{
    reply.send([extreme,minor]);
  } catch (error) {
    console.error(error);
    reply.status(500).send({ error: "Internal Server Error" });
  }
});

fastify.get("/code", async (req, reply) => {
  try{
    reply.send(req.body.plot); // Send the API response back to the client
  } catch (error) {
    console.error(error);
    reply.status(500).send({ error: "Internal Server Error" });
  }
});


// Start the server
fastify.listen({ port: 3000 }, (err, address) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(`Server running at ${address}`);
});
