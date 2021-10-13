import http from "http";
import express from "express";
import SocketIO from "socket.io";


const app = express();

app.set("view engine", "pug");
app.set("views", __dirname + "/views");
app.use("/public", express.static(__dirname + "/public"));
app.get("/", (_, res) => res.render("home"));
app.get("/*", (_, res) => res.redirect("/"));

const handleListen = () => console.log(`Listening on http://localhost:3000`);

const httpServer = http.createServer(app);
const wsServer = SocketIO(httpServer);

function publicRooms() {
    const {
        sockets: {
          adapter: { sids, rooms },
        },
    } = wsServer;
    const publicRooms = [];
    rooms.forEach((_, key) => {
        if (sids.get(key) === undefined) {
            publicRooms.push(key);
        }
    });
    return publicRooms;
}
    
function countRoom(roomName) {
    return wsServer.sockets.adapter.rooms.get(roomName)?.size;
}

wsServer.on("connection", (socket) => {
    socket.onAny((event) => {
      console.log(`Socket Event: ${event}`);
    });
  
    socket.on("nickname", (nickname) => (socket["nickname"] = nickname));
  
    socket.on("get_room", () => {
      socket.emit("get_room", publicRooms());
    });
  
    socket.on("enter_room", (roomName) => {
      socket.join(roomName);
      socket.to(roomName).emit("enter_room");
      wsServer
        .to(roomName)
        .emit("enter_room_all", socket.nickname, countRoom(roomName));
      wsServer.sockets.emit("room_change", publicRooms());
    });
  
    socket.on("offer", (offer, roomName) => {
      socket.to(roomName).emit("offer", offer);
    });
  
    socket.on("answer", (answer, roomName) => {
      socket.to(roomName).emit("answer", answer);
    });
  
    socket.on("ice", (ice, roomName) => {
      socket.to(roomName).emit("ice", ice);
    });
  
    socket.on("new_message", (message, room, done) => {
      socket.to(room).emit("new_message", `${socket.nickname}: ${message}`);
      done();
    });
  
    socket.on("disconnecting", () => {
      socket.rooms.forEach((room) => {
        socket.to(room).emit("leave_room", socket.nickname, countRoom(room) - 1);
      });
    });
  
    socket.on("disconnect", () => {
      wsServer.sockets.emit("room_change", publicRooms());
    });
});
  
httpServer.listen(3000, handleListen);